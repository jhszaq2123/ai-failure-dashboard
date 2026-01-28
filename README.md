# AI Machine Failure Dashboard

Interactive Streamlit dashboard for predicting machine failures based on sensor data.

## Features
- 7 industrial sensor parameters (Temperature, Vibration, Pressure, Humidity, Load, RPM, Voltage)
- Machine learning failure prediction (RandomForest)
- Interactive dashboard using Streamlit + Plotly
- Manual sensor input with real-time prediction

## Model
- RandomForestClassifier (scikit-learn)
- Binary classification: failure / no failure

## Usage

### 1. Create virtual environment
```bash
python -m venv venv
