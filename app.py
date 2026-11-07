from pathlib import Path

light_app = r'''import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Today’s Weather Forecast",
    page_icon="🌦️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Today’s Weather Forecast 🌦️")
st.write(
    "Enter air temperature, relative humidity, and wind speed "
    "to predict the expected rainfall category."
)

temperature = st.slider(
    "Air Temperature (°C)", 20.0, 40.0, 30.0, 0.1
)

humidity = st.slider(
    "Relative Humidity (%)", 50, 100, 80, 1
)

wind_speed = st.slider(
    "Wind Speed (m/s)", 0.0, 10.0, 2.0, 0.1
)

if st.button("Predict Weather", type="primary"):
    X = pd.DataFrame(
        [[temperature, humidity, wind_speed]],
        columns=["suhu", "kelembaban", "kecepatan_angin"]
    )

    prediction = model.predict(X)[0]

    labels = {
        "Hujan Lebat": "🌧️ Heavy Rainfall",
        "Hujan Ringan": "🌦️ Light Rainfall",
        "Tidak Hujan": "☀️ No Rainfall"
    }

    st.subheader("Prediction Result")
    st.success(labels.get(prediction, str(prediction)))

    # Show probabilities only when supported by the model
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        probability_df = pd.DataFrame({
            "Rainfall Category": [
                labels.get(c, str(c)) for c in model.classes_
            ],
            "Probability (%)": (probabilities * 100).round(2)
        })
        st.subheader("Prediction Probabilities")
        st.dataframe(
            probability_df,
            hide_index=True,
            use_container_width=True
        )

st.caption(
    "Model: Random Forest Classifier | "
    "Inputs: air temperature, relative humidity, wind speed"
)
'''

out = Path("/mnt/data/app_light.py")
out.write_text(light_app, encoding="utf-8")
print(f"Created: {out}")
print(f"Size: {out.stat().st_size:,} bytes")
