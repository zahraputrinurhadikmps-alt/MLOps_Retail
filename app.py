from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model_pembelian.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(
        page_title="Prediksi Penjualan Ritel",
        page_icon="🛒",
        layout="wide",
    )

    model = load_model()

    st.title("🛒 Prediksi Penjualan Ritel")
    st.write("Machine Learning Deployment menggunakan Streamlit")

    col1, col2 = st.columns(2)

    with col1:
        sale_date = st.date_input("Tanggal Penjualan")
        sale_time = st.time_input("Waktu Penjualan")
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age = st.number_input("Usia", min_value=1, max_value=100, value=25)

    with col2:
        category = st.selectbox("Kategori", ["Clothing", "Beauty", "Electronics"])
        quantiy = st.number_input("Kuantitas", min_value=1, value=1)
        price_per_unit = st.number_input("Harga Per Unit", min_value=0.0, value=100.0)
        cogs = st.number_input("COGS", min_value=0.0, value=80.0)

    if st.button("Prediksi"):
        try:
            data = pd.DataFrame([{
                "sale_date": sale_date.strftime("%Y-%m-%d"),
                "sale_time": sale_time.strftime("%H:%M:%S"),
                "gender": gender,
                "age": age,
                "category": category,
                "quantiy": quantiy,
                "price_per_unit": price_per_unit,
                "cogs": cogs,
            }])

            hasil = model.predict(data)
            prediksi = round(float(hasil[0]), 2)

            st.success("Prediksi berhasil")
            st.metric("Hasil Prediksi", f"Rp {prediksi:,.2f}")
        except Exception as exc:
            st.error(f"Terjadi kesalahan saat memprediksi: {exc}")


if __name__ == "__main__":
    main()