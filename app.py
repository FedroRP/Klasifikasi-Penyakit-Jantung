
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('best_random_forest_model.pkl')

# Title
st.title("Prediksi Penyakit Jantung")
st.write("Masukkan data pasien untuk memprediksi apakah memiliki penyakit jantung.")

# User input (13 fitur)
age = st.number_input("Umur", min_value=1, max_value=120, value=50)
sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
cp = st.selectbox("Tipe Nyeri Dada (CP)", [0, 1, 2, 3])
trestbps = st.number_input("Tekanan Darah Saat Istirahat", min_value=80, max_value=200, value=120)
chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=240)
fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl", [0, 1])
restecg = st.selectbox("Hasil EKG Saat Istirahat", [0, 1, 2])
thalach = st.number_input("Detak Jantung Maksimum", min_value=60, max_value=250, value=150)
exang = st.selectbox("Nyeri Dada Saat Olahraga (Angina)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Jumlah Pembuluh Utama (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2])

# Convert input
sex_val = 1 if sex == "Laki-laki" else 0

# Buat array input
input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Prediction
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    result = "Punya penyakit jantung" if prediction[0] == 1 else "Tidak punya penyakit jantung"
    st.success(f"Hasil Prediksi: {result}")
