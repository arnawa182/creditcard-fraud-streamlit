import pandas as pd
import numpy as np
import random
import logging
import datetime
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.combine import SMOTETomek
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from deap import base, creator, tools, algorithms
from skopt import gp_minimize, load, dump
from skopt.space import Integer, Real
from skopt.utils import use_named_args

BO_SAVE_PATH = "bo_result.pkl"

def preprocess_data(filepath='creditcard.csv'):
    df = pd.read_csv(filepath)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X = SimpleImputer(strategy='median').fit_transform(X)
    X = StandardScaler().fit_transform(X)
    smote = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )
    return (
        X_train.astype(np.float32),
        y_train.to_numpy().astype(np.float32),
        X_test.astype(np.float32),
        y_test.to_numpy().astype(np.float32)
    )

def create_model(n1, n2, dropout1, dropout2, lr, input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(n1, activation='relu'),
        layers.Dropout(dropout1),
        layers.Dense(n2, activation='relu'),
        layers.Dropout(dropout2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model_cv(params, X, y, n_splits=3):
    if len(params) != 5:
        return 1.0
    n1, n2, d1, d2, lr = int(params[0]), int(params[1]), float(params[2]), float(params[3]), float(params[4])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        model = create_model(n1, n2, d1, d2, lr, X.shape[1])
        model.fit(X_train_fold, y_train_fold,
                  validation_data=(X_val_fold, y_val_fold),
                  epochs=30, batch_size=64,
                  callbacks=[
                      EarlyStopping(patience=5, restore_best_weights=True),
                      ReduceLROnPlateau(patience=3, factor=0.5)
                  ],
                  verbose=0)
        y_pred = model.predict(X_val_fold).ravel()
        f1 = f1_score(y_val_fold, y_pred > 0.5, pos_label=1)
        f1_scores.append(f1)
    return 1 - np.mean(f1_scores)

space = [
    Integer(64, 128, name='n1'),
    Integer(32, 64, name='n2'),
    Real(0.1, 0.3, name='dropout1'),
    Real(0.1, 0.3, name='dropout2'),
    Real(1e-5, 1e-3, prior='log-uniform', name='lr')
]

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_n1", random.randint, 64, 128)
toolbox.register("attr_n2", random.randint, 32, 64)
toolbox.register("attr_d1", random.uniform, 0.1, 0.3)
toolbox.register("attr_d2", random.uniform, 0.1, 0.3)
toolbox.register("attr_lr", random.uniform, 1e-5, 1e-3)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_n1, toolbox.attr_n2, toolbox.attr_d1, toolbox.attr_d2, toolbox.attr_lr), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def ga_eval(ind, X, y):
    return (evaluate_model_cv(ind, X, y),)

def custom_mutation(ind, indpb):
    if random.random() < indpb: ind[0] = random.randint(64, 128)
    if random.random() < indpb: ind[1] = random.randint(32, 64)
    if random.random() < indpb: ind[2] = random.uniform(0.1, 0.3)
    if random.random() < indpb: ind[3] = random.uniform(0.1, 0.3)
    if random.random() < indpb: ind[4] = random.uniform(1e-5, 1e-3)
    return (ind,)

def run_hybrid_ga(X, y, bo_params, ngen=20):
    toolbox.register("evaluate", ga_eval, X=X, y=y)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", custom_mutation, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    population = toolbox.population(n=10)
    population[0][:] = bo_params
    hof = tools.HallOfFame(1)
    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.4)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        population[:] = toolbox.select(offspring, k=len(population))
        hof.update(population)
    return hof[0]

def main_pipeline(filepath='creditcard.csv'):
    X_train, y_train, X_test, y_test = preprocess_data(filepath)

    @use_named_args(space)
    def bo_objective(**params):
        return evaluate_model_cv([params['n1'], params['n2'], params['dropout1'], params['dropout2'], params['lr']], X_train, y_train)

    if os.path.exists(BO_SAVE_PATH):
        result_bo = load(BO_SAVE_PATH)
    else:
        result_bo = gp_minimize(bo_objective, space, n_calls=20, random_state=42)
        dump(result_bo, BO_SAVE_PATH)

    bo_best = result_bo.x
    best_ind = run_hybrid_ga(X_train, y_train, bo_best)

    model = create_model(int(best_ind[0]), int(best_ind[1]), float(best_ind[2]), float(best_ind[3]), float(best_ind[4]), X_train.shape[1])
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                         ReduceLROnPlateau(patience=3)], verbose=0)

    y_pred = model.predict(X_test).ravel()
    y_label = (y_pred > 0.5).astype(int)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_path = f"hasil_evaluasi_hybrid_GA_BO_{ts}.txt"
    roc_img = f"roc_curve_hybrid_GA_BO_{ts}.png"
    pr_img = f"pr_curve_hybrid_GA_BO_{ts}.png"

    with open(txt_path, "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(classification_report(y_test, y_label))
        f.write("\n=== Confusion Matrix ===\n")
        f.write(str(confusion_matrix(y_test, y_label)))
        f.write(f"\nAUC Score: {roc_auc_score(y_test, y_pred):.4f}\n")
        f.write(f"Best Params: {best_ind}\n")

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(roc_img)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.figure()
    plt.plot(recall, precision, label=f"PR (AP={average_precision_score(y_test, y_pred):.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(pr_img)
    plt.close()

    return txt_path, roc_img, pr_img
