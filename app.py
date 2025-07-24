import streamlit as st
from main_pipeline import main_pipeline

st.title("Deteksi Penipuan Kartu Kredit")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    if st.button("Jalankan Pipeline"):
        with open("creditcard.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.text("Menjalankan pipeline... Mohon tunggu beberapa menit.")
        txt_path, roc_img, pr_img = main_pipeline("creditcard.csv")

        st.success("Selesai! Berikut hasilnya:")

        with open(txt_path, "r") as f:
            report = f.read()
        st.text_area("Hasil Evaluasi", value=report, height=300)

        st.image(roc_img, caption="ROC Curve")
        st.image(pr_img, caption="Precision-Recall Curve")
else:
    st.warning("Silakan unggah file CSV terlebih dahulu.")
