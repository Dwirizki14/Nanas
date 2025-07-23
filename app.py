import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model dan scaler
model = joblib.load("model_knn_clean.pkl")
scaler = joblib.load("scaler_knn_clean.pkl")

st.title("Klasifikasi Tingkat Kematangan Buah Nanas 🍍")

st.write("Silakan input nilai RGB kulit buah nanas:")

# Input RGB
r = st.slider("Red (R)", 0, 255, 100)
g = st.slider("Green (G)", 0, 255, 100)
b = st.slider("Blue (B)", 0, 255, 100)

if st.button("Prediksi"):
    rgb_input = np.array([[r, g, b]])
    scaled_input = scaler.transform(rgb_input)
    pred = model.predict(scaled_input)[0]

    # Label mapping (jika pakai label encoder, sesuaikan)
    label_map = {0: "Mentah", 1: "Setengah Matang", 2: "Matang"}
    hasil = label_map.get(pred, "Tidak diketahui")

    st.success(f"Hasil Prediksi: **{hasil}**")

    # Warna RGB preview
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    st.markdown(f"<div style='width:100px;height:50px;background-color:{hex_color};border-radius:10px;border:1px solid #000;'></div>", unsafe_allow_html=True)
