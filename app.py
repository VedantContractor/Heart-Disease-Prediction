import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.markdown(
    """
    <style>
        :root {
            --bg-1: #07111f;
            --bg-2: #101c34;
            --panel: rgba(15, 23, 42, 0.72);
            --panel-border: rgba(148, 163, 184, 0.22);
            --primary: #38bdf8;
            --primary-strong: #2563eb;
            --accent: #f97316;
            --success: #34d399;
            --danger: #f87171;
            --text: #e2e8f0;
            --muted: #cbd5e1;
            --shadow: rgba(14, 116, 144, 0.28);
        }

        .stApp {
            background: linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 45%, #172554 100%);
            color: var(--text);
        }

        .stApp > div {
            background: transparent;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        h1, h2, h3 {
            letter-spacing: 0.02em;
            line-height: 1.2;
        }

        .hero-section {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.14), rgba(168, 85, 247, 0.12), rgba(15, 118, 110, 0.12));
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 20px;
            padding: 1.5rem 1.6rem 1.1rem 1.6rem;
            box-shadow: 0 18px 45px rgba(14, 116, 144, 0.18);
            margin-bottom: 1.4rem;
        }

        .hero-title {
            color: #f8fafc;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 0.4rem;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1.02rem;
            margin: 0;
        }

        .section-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.62));
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 1.1rem 1rem 0.5rem 1rem;
            box-shadow: 0 10px 25px rgba(59, 130, 246, 0.08);
            margin-bottom: 1.1rem;
        }

        .result-box {
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.18), rgba(59, 130, 246, 0.12));
            border: 1px solid rgba(45, 212, 191, 0.25);
            border-radius: 18px;
            padding: 1.2rem 1rem;
            margin-top: 1rem;
            box-shadow: 0 16px 38px rgba(16, 185, 129, 0.10);
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--primary-strong));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 1.2rem;
            font-weight: 700;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.28);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(37, 99, 235, 0.35);
        }

        div[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.82);
            border-right: 1px solid rgba(148, 163, 184, 0.15);
        }

        .stSelectbox label, .stNumberInput label, .stSlider label {
            color: #e2e8f0 !important;
            font-weight: 600;
        }

        .stAlert {
            border-radius: 14px;
        }

        .stProgress > div > div {
            background: linear-gradient(90deg, #34d399, #22c55e, #14b8a6);
        }

        @media (max-width: 768px) {
            .hero-title {
                font-size: 2rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_files():

    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")

    return model, scaler, expected_columns


model, scaler, expected_columns = load_files()


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">❤️ Heart Disease Prediction</div>
        <p class="hero-subtitle">Enter the patient's health information below to estimate the likelihood of heart disease using the trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.header("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider(
        "Age",
        min_value=1,
        max_value=100,
        value=40
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

with col3:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )


# =========================================================
# HEART INFORMATION
# =========================================================

st.header("🫀 Heart & Blood Information")

col1, col2, col3 = st.columns(3)

with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

with col2:
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120
    )

with col3:
    cholesterol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=200
    )


# =========================================================
# ECG & EXERCISE INFORMATION
# =========================================================

st.header("🏃 ECG & Exercise Information")

col1, col2, col3 = st.columns(3)

with col1:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

with col2:
    max_hr = st.slider(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )

with col3:
    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["Y", "N"]
    )


# =========================================================
# ST SEGMENT
# =========================================================

st.header("📈 ST Segment Information")

col1, col2 = st.columns(2)

with col1:
    oldpeak = st.slider(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

with col2:
    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict = st.button(
    "🔍 Predict Heart Disease",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    # -----------------------------------------------------
    # CREATE RAW INPUT
    # -----------------------------------------------------

    raw_input = {

        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,

        "ChestPainType_" + chest_pain: 1,

        "RestingECG_" + resting_ecg: 1,

        "ExerciseAngina_" + exercise_angina: 1,

        "ST_Slope_" + st_slope: 1
    }


    # -----------------------------------------------------
    # CREATE DATAFRAME
    # -----------------------------------------------------

    input_df = pd.DataFrame([raw_input])


    # -----------------------------------------------------
    # ADD MISSING COLUMNS
    # -----------------------------------------------------

    for col in expected_columns:

        if col not in input_df.columns:

            input_df[col] = 0


    # -----------------------------------------------------
    # REORDER COLUMNS
    # -----------------------------------------------------

    input_df = input_df[expected_columns]


    # -----------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------

    scaled_input = scaler.transform(input_df)


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(scaled_input)[0]


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    probability = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(scaled_input)[0][1]


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.divider()

    st.header("📊 Prediction Result")

    if prediction == 1:
        risk_title = "⚠️ HIGH RISK OF HEART DISEASE"
        risk_message = "The model predicts a higher likelihood of heart disease."
        risk_color = "#fca5a5"
        accent = "rgba(248, 113, 113, 0.22)"
    else:
        risk_title = "✅ LOW RISK OF HEART DISEASE"
        risk_message = "The model predicts a lower likelihood of heart disease."
        risk_color = "#86efac"
        accent = "rgba(52, 211, 153, 0.22)"

    st.markdown(
        f"""
        <div class="result-box" style="background: linear-gradient(135deg, {accent}, rgba(59, 130, 246, 0.08)); border-color: {risk_color};">
            <h3 style="color: {risk_color}; margin: 0 0 0.45rem 0;">{risk_title}</h3>
            <p style="margin: 0; color: #e2e8f0;">{risk_message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    if probability is not None:

        st.subheader("Prediction Probability")

        percentage = probability * 100

        st.progress(float(probability))

        st.write(
            f"Estimated probability of heart disease: "
            f"**{percentage:.2f}%**"
        )


    # -----------------------------------------------------
    # SHOW INPUT DATA
    # -----------------------------------------------------

    with st.expander("🔎 View Entered Information"):

        st.write({
            "Age": age,
            "Sex": sex,
            "Chest Pain Type": chest_pain,
            "Resting BP": resting_bp,
            "Cholesterol": cholesterol,
            "Fasting Blood Sugar": fasting_bs,
            "Resting ECG": resting_ecg,
            "Maximum Heart Rate": max_hr,
            "Exercise Angina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST Slope": st_slope
        })


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "⚠️ This application is developed for educational and "
    "machine-learning demonstration purposes only. "
    "It is not a medical diagnosis tool."
) 
