
import streamlit as st
import numpy as np
import joblib
import base64

# ================= BACKGROUND IMAGE =================
def set_background(png_file):
    with open(png_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        /* Make all text white */
        .stMarkdown, .stRadio, .stNumberInput, .stSelectbox, label, p, h1, h2, h3, h4, h5, h6 {{
            color: white !important;
        }}
        /* Make radio options white */
        div[role="radiogroup"] > label > div[data-testid="stMarkdownContainer"] p {{
            color: white !important;
            font-weight: bold;
        }}
        /* Style prediction card */
        .result-card {{
            background-color: rgba(0, 0, 50, 0.7);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: white;
            margin-top: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ================= LOAD MODEL =================
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="🧬 Hepatitis Survival Prediction App", page_icon="🩺", layout="centered")
set_background("background.png")

# ================= HEADER =================
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo.png", width=90)  # hospital logo
with col2:
    st.markdown("<h1 style='color:white;'>🧬 Hepatitis Survival Prediction</h1>", unsafe_allow_html=True)
st.markdown("This app predicts whether a patient with hepatitis is likely to **survive** or **not survive** based on clinical data.", unsafe_allow_html=True)

# ================= INPUT FORM =================
st.markdown("<h2 style='color:white;'>📋 Enter Patient Details</h2>", unsafe_allow_html=True)

age = st.number_input("🎂 Age", min_value=7, max_value=100, value=30)
sex = st.radio("⚧ Sex", ["Male", "Female"])
steroid = st.radio("💊 Steroid", ["Yes", "No"])
antivirals = st.radio("🧪 Antivirals", ["Yes", "No"])
fatigue = st.radio("😴 Fatigue", ["Yes", "No"])
malaise = st.radio("🤒 Malaise", ["Yes", "No"])
anorexia = st.radio("🥗 Anorexia", ["Yes", "No"])
liver_big = st.radio("🫀 Liver Big", ["Yes", "No"])
liver_firm = st.radio("🧩 Liver Firm", ["Yes", "No"])
spleen_palpable = st.radio("🫁 Spleen Palpable", ["Yes", "No"])
spiders = st.radio("🕷️ Spiders", ["Yes", "No"])
ascites = st.radio("💧 Ascites", ["Yes", "No"])
varices = st.radio("🩸 Varices", ["Yes", "No"])
bilirubin = st.number_input("🧫 Bilirubin", min_value=0.0, value=1.0)
alk_phosphate = st.number_input("⚗️ Alk Phosphate", min_value=0.0, value=80.0)
sgot = st.number_input("🧬 SGOT", min_value=0.0, value=20.0)
albumin = st.number_input("🧑‍⚕️ Albumin", min_value=0.0, value=4.0)
protime = st.number_input("⏱️ Protime", min_value=0.0, value=60.0)
histology = st.radio("🔬 Histology", ["Yes", "No"])

# ================= ENCODER =================
def encode(val):
    return 1 if val in ["Yes", "Male"] else 2 if val == "Female" else 0

inputs = [
    age, encode(sex), encode(steroid), encode(antivirals), encode(fatigue),
    encode(malaise), encode(anorexia), encode(liver_big), encode(liver_firm),
    encode(spleen_palpable), encode(spiders), encode(ascites), encode(varices),
    bilirubin, alk_phosphate, sgot, albumin, protime, encode(histology)
]

# ================= PREDICTION =================
if st.button("🔍 Predict"):
    X = scaler.transform([inputs])
    prediction = rf_model.predict(X)[0]

    if prediction == 1:
        st.markdown("<div class='result-card'>✅ The patient is <b>LIKELY TO SURVIVE</b>.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-card'>⚠️ The patient is <b>NOT LIKELY TO SURVIVE</b>.</div>", unsafe_allow_html=True)
