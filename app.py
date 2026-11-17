import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Predictor de Severidad", page_icon="🛡️", layout="wide")
st.title("🛡️ Predicción de Vulnerabilidades en Software")

MODEL_PATH = "vulnerability_model.joblib"
if not os.path.exists(MODEL_PATH):
    st.error("Ejecuta primero 'python entrenar_modelo.py' para generar el archivo 'vulnerability_model.joblib'.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
le = bundle["label_encoder"]
num_features = bundle["num_features"]
cat_features = bundle["cat_features"]
feature_ranges = bundle["feature_ranges"]
cat_options = bundle["cat_options"]
metrics_df = pd.DataFrame(bundle["metrics_summary"])

st.sidebar.header("Parámetros de Entrada")
user_input = {}

for col in num_features:
    min_v, max_v, med_v = feature_ranges[col]
    if min_v == max_v:
        max_v = min_v + 10.0
    user_input[col] = st.sidebar.number_input(col, min_value=float(min_v), max_value=float(max_v * 1.5), value=float(med_v))

for col in cat_features:
    opts = cat_options.get(col) or ["Desconocido"]
    user_input[col] = st.sidebar.selectbox(col, opts)

tab1, tab2 = st.tabs(["Predicción", "Comparativa de Modelos"])

with tab1:
    input_df = pd.DataFrame([user_input])
    pred_idx = pipeline.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    st.metric(label=f"Severidad Predicha ({bundle['best_model_name']})", value=str(pred_label))

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        prob_df = pd.DataFrame({"Severidad": le.classes_, "Probabilidad (%)": np.round(probs * 100, 2)})
        fig = px.bar(prob_df, x="Probabilidad (%)", y="Severidad", orientation="h", text="Probabilidad (%)")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(metrics_df, use_container_width=True)
