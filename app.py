```python
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Title
st.title("Today’s Weather Forecast 🌦️")
st.write(
    "Enter temperature, humidity, and wind speed data "
    "to predict whether it will rain today."
)

# Manual user input
suhu = st.slider("Air Temperature (°C)", 20.0, 40.0, 30.0)
kelembaban = st.slider("Relative Humidity (%)", 50, 100, 80)
angin = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)

# Prediction
if st.button("Predict"):
    data = pd.DataFrame(
        [[suhu, kelembaban, angin]],
        columns=["suhu", "kelembaban", "kecepatan_angin"]
    )

    prediksi = model.predict(data)[0]

    st.subheader("Prediction Result:")

    if prediksi == "Hujan Lebat":
        st.error("🌧️ Heavy Rainfall")
    elif prediksi == "Hujan Ringan":
        st.warning("🌦️ Light Rainfall")
    else:
        st.success("☀️ No Rainfall")
```
