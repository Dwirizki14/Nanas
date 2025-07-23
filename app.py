import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("model_knn_clean.pkl")
scaler = joblib.load("scaler_knn_clean.pkl")

# Judul aplikasi
st.title("🍍 Klasifikasi Kematangan Nanas Berdasarkan Warna Kulit")

st.markdown("""
Masukkan nilai RGB (Red, Green, Blue) dari gambar kulit nanas untuk memprediksi tingkat kematangan:
""")

# Input nilai RGB dari user
r = st.slider("Red (R)", 0, 255, 128)
g = st.slider("Green (G)", 0, 255, 128)
b = st.slider("Blue (B)", 0, 255, 128)

# Tampilkan warna yang dipilih
hex_color = '#%02x%02x%02x' % (r, g, b)
st.markdown(
    f"<div style='width:100px;height:50px;background-color:{hex_color};border-radius:10px;border:1px solid #000;'></div>",
    unsafe_allow_html=True
)

# Prediksi saat tombol ditekan
if st.button("Prediksi Kematangan"):
    input_rgb = np.array([[r, g, b]])
    input_scaled = scaler.transform(input_rgb)
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"🔮 Prediksi: **{prediction}**")
