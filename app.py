import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('random_forest_heart.pkl')
scaler = joblib.load('scaler.pkl')  # pastikan scaler.pkl berada di folder yang sama

st.title("Prediksi Penyakit Jantung")

# Form input fitur
age = st.number_input('Umur', min_value=1, max_value=120, value=50)
sex = st.selectbox('Jenis Kelamin', options=[0, 1], format_func=lambda x: 'Perempuan' if x == 0 else 'Laki-laki')
cp = st.selectbox('Tipe Nyeri Dada (cp)', options=[0, 1, 2, 3])
trestbps = st.number_input('Tekanan Darah Istirahat (trestbps)', min_value=80, max_value=200, value=120)
chol = st.number_input('Kolesterol (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Gula Darah Puasa > 120 mg/dl (fbs)', options=[0, 1])
restecg = st.selectbox('Hasil Elektrokardiografi Istirahat (restecg)', options=[0, 1, 2])
thalach = st.number_input('Detak Jantung Maksimum (thalach)', min_value=60, max_value=220, value=150)
exang = st.selectbox('Induksi Angina (exang)', options=[0, 1])
oldpeak = st.number_input('Depresi ST (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox('Kemiringan ST (slope)', options=[0, 1, 2])
ca = st.selectbox('Jumlah Vessels (ca)', options=[0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (thal)', options=[0, 1, 2, 3])

# Saat tombol ditekan
if st.button('Prediksi'):
    # Siapkan fitur dalam bentuk array 2D
    input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]])

    # Lakukan scaling
    scaled_input = scaler.transform(input_features)

    # Prediksi
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]  # Probabilitas kelas 1 (sakit)

    st.write(f"Probabilitas memiliki penyakit jantung: {probability:.2f}")

    if prediction == 1:
        st.error("Pasien kemungkinan memiliki penyakit jantung.")
    else:
        st.success("Pasien kemungkinan TIDAK memiliki penyakit jantung.")
