import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Today’s Weather Forecast",
    page_icon="🌦️",
    layout="centered"
)

# ---------------------------------------------------------
# Load the trained Random Forest model
# ---------------------------------------------------------
model = joblib.load("model.pkl")

# ---------------------------------------------------------
# Application title and scientific description
# ---------------------------------------------------------
st.title("Today’s Weather Forecast 🌦️")

st.markdown(
    "This web-based application uses a trained **Random Forest classifier** "
    "to estimate the expected rainfall category from three meteorological "
    "variables: air temperature, relative humidity, and wind speed."
)

st.info(
    "Enter the current meteorological conditions below. "
    "The model will classify the expected rainfall condition."
)

# ---------------------------------------------------------
# User input
# ---------------------------------------------------------
st.subheader("Meteorological Inputs")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider(
        "Air Temperature (°C)",
        min_value=20.0,
        max_value=40.0,
        value=30.0,
        step=0.1,
        help="Air temperature in degrees Celsius."
    )

with col2:
    humidity = st.slider(
        "Relative Humidity (%)",
        min_value=50,
        max_value=100,
        value=80,
        step=1,
        help="Relative humidity expressed as a percentage."
    )

wind_speed = st.slider(
    "Wind Speed (m/s)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Wind speed in metres per second."
)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("Predict Weather", type="primary", use_container_width=True):

    # IMPORTANT:
    # These feature names must remain identical to those used
    # when the model was trained.
    input_data = pd.DataFrame(
        [[temperature, humidity, wind_speed]],
        columns=["suhu", "kelembaban", "kecepatan_angin"]
    )

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Map probabilities to model classes
    probability_table = pd.DataFrame({
        "Rainfall Category": model.classes_,
        "Probability (%)": probabilities * 100
    }).sort_values("Probability (%)", ascending=False)

    # -----------------------------------------------------
    # Prediction result
    # -----------------------------------------------------
    st.subheader("Prediction Result")

    if prediction == "Hujan Lebat":
        st.error("🌧️ **Heavy Rainfall**")
        result_description = (
            "The model predicts a heavy rainfall condition "
            "under the entered meteorological conditions."
        )

    elif prediction == "Hujan Ringan":
        st.warning("🌦️ **Light Rainfall**")
        result_description = (
            "The model predicts a light rainfall condition "
            "under the entered meteorological conditions."
        )

    else:
        st.success("☀️ **No Rainfall**")
        result_description = (
            "The model predicts no rainfall under the entered "
            "meteorological conditions."
        )

    st.write(result_description)

    # -----------------------------------------------------
    # Probability distribution
    # -----------------------------------------------------
    st.subheader("Prediction Probabilities")

    display_table = probability_table.copy()
    display_table["Rainfall Category"] = display_table[
        "Rainfall Category"
    ].replace({
        "Hujan Lebat": "Heavy Rainfall",
        "Hujan Ringan": "Light Rainfall",
        "Tidak Hujan": "No Rainfall"
    })

    display_table["Probability (%)"] = display_table[
        "Probability (%)"
    ].round(2)

    st.dataframe(
        display_table,
        hide_index=True,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Input summary
    # -----------------------------------------------------
    st.subheader("Input Conditions")

    summary = pd.DataFrame({
        "Meteorological Variable": [
            "Air Temperature",
            "Relative Humidity",
            "Wind Speed"
        ],
        "Value": [
            f"{temperature:.1f} °C",
            f"{humidity:d} %",
            f"{wind_speed:.1f} m/s"
        ]
    })

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

# ---------------------------------------------------------
# Scientific note
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Scientific note: This application provides model-based rainfall "
    "classification and should be interpreted as a decision-support output, "
    "not as an official meteorological warning."
)

st.caption(
    "Model: Random Forest Classifier | "
    "Input variables: air temperature, relative humidity, wind speed"
)
