import streamlit as st
import pandas as pd
import joblib
from datetime import date, datetime

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="🛒 Prediksi Total Penjualan Retail",
    page_icon="🛒"
)


# ==========================
# Load Model
# ==========================
pipeline = joblib.load("model_pembelian.pkl")

# ==========================
# Judul
# ==========================
st.title(" 🛒Prediksi Penjualan Retail")
st.write("Machine Learning Deployment menggunakan Streamlit")

st.divider()

# ==========================
# Input
# ==========================

tanggal = st.date_input(
    "Tanggal Penjualan",
    value=date.today()
)

waktu = st.time_input(
    "Waktu Penjualan",
    value=datetime.now().time()
)

gender = st.selectbox(
    "Jenis Kelamin",
    ["Male", "Female"]
)

age = st.number_input(
    "Usia",
    min_value=18,
    max_value=80,
    value=25
)

category = st.selectbox(
    "Kategori",
    [
        "Beauty",
        "Clothing",
        "Electronics"
    ]
)

quantiy = st.number_input(
    "Kuantitas",
    min_value=1,
    value=1
)

price = st.number_input(
    "Harga Per Unit",
    min_value=1.0,
    value=100.0
)

cogs = st.number_input(
    "COGS",
    min_value=1.0,
    value=80.0
)

# ==========================
# Ambil Bulan & Jam
# ==========================

month = tanggal.month
hour = waktu.hour

# ==========================
# Tombol
# ==========================

if st.button("Prediksi", use_container_width=True):

    data = pd.DataFrame({
        "gender":[gender],
        "age":[age],
        "category":[category],
        "quantiy":[quantiy],
        "cogs":[cogs],
        "month":[month],
        "hour":[hour]
    })

    hasil = pipeline.predict(data)[0]

    st.divider()

    st.subheader("Hasil Prediksi")

    st.metric(
        "Prediksi Total Penjualan",
        f"Rp {hasil:,.2f}"
    )