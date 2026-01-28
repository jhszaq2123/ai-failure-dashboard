import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ======================
# 1️⃣ Streamlit setup
# ======================
st.set_page_config(page_title="AI Machine Failure Dashboard", layout="wide")
st.title("📊 AI Machine Failure Dashboard")

# ======================
# 2️⃣ Load Data & Model
# ======================
df = pd.read_csv("data/sensor_data.csv")
model = joblib.load("models/model.pkl")

sensors = ["temperature","vibration","pressure","humidity","load","rpm","voltage"]

# Dodanie predykcji AI
df["pred_prob"] = model.predict_proba(df[sensors])[:,1]
df["pred_class"] = model.predict(df[sensors])

# ======================
# 3️⃣ Sidebar – manual input
# ======================
st.sidebar.header("Manual Sensor Input")
sensor_values = {}
for s in sensors:
    sensor_values[s] = st.sidebar.slider(s.capitalize(),
                                         float(df[s].min()),
                                         float(df[s].max()),
                                         float(df[s].mean()))

input_df = pd.DataFrame([list(sensor_values.values())], columns=sensors)
pred_sidebar = model.predict(input_df)[0]
prob_sidebar = model.predict_proba(input_df)[0][1]

st.sidebar.subheader("🤖 Sidebar Prediction")
if pred_sidebar == 1:
    st.sidebar.error(f"⚠️ Failure predicted! Probability: {prob_sidebar:.2f}")
else:
    st.sidebar.success(f"✅ No failure predicted. Probability: {prob_sidebar:.2f}")

st.sidebar.progress(int(prob_sidebar*100))

# ======================
# 4️⃣ KPI – aktualne wartości sensorów
# ======================
st.subheader("📌 Current Sensor Values")
cols = st.columns(4)
for i, s in enumerate(sensors):
    cols[i % 4].metric(s.capitalize(), sensor_values[s])
cols[3].metric("Predicted Failure Probability", f"{prob_sidebar:.2f}")

# ======================
# 5️⃣ Wykresy dla każdego sensora
# ======================
st.subheader("📈 Sensor Data Visualization")

for s in sensors:
    fig = px.histogram(df, x=s, nbins=30, color="pred_class",
                       color_discrete_map={0:"green", 1:"red"},
                       title=f"{s.capitalize()} Distribution by Failure")
    fig.update_layout(yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
