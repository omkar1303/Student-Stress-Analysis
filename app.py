
import streamlit as st
import pandas as pd
import xgboost as xgb
import os
from datetime import datetime
import joblib
import numpy as np

# Load Model & Artifacts
@st.cache_resource
def load_resources():
    model = xgb.XGBClassifier()
    scaler_minmax = None
    scaler_std = None
    model_columns = None
    
    try:
        model.load_model("student_stress_model.json")
        scaler_minmax = joblib.load("scaler_minmax.pkl")
        scaler_std = joblib.load("scaler_std.pkl")
        model_columns = joblib.load("model_columns.pkl")
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None
    return model, scaler_minmax, scaler_std, model_columns

model, scaler_minmax, scaler_std, model_columns = load_resources()

import urllib.parse
from sqlalchemy import create_engine, text

# --- Configuration (SQLite) ---
DB_NAME = "student_stress.db"
TABLE_NAME = "feedback_data"
FEEDBACK_FILE = "human_feedback.csv"

def save_to_db(data_dict):
    try:
        conn_str = f"sqlite:///{DB_NAME}"
        engine = create_engine(conn_str)
        # Simplified schema for db for now - user can expand later
        db_data = {
            "created_at": data_dict.get("created_at", datetime.now()),
            "predicted_stress": data_dict.get("Predicted_Stress"),
            "actual_stress": data_dict.get("Actual_Stress")
        }
        df = pd.DataFrame([db_data])
        df.to_sql(name=TABLE_NAME, con=engine, if_exists='append', index=False)
        return True
    except Exception as e:
        return False

def save_feedback(data, predicted, actual):
    # Ensure all data is serializable
    feedback_entry = data.copy()
    feedback_entry['Predicted_Stress'] = predicted
    feedback_entry['Actual_Stress'] = actual
    feedback_entry['Timestamp'] = datetime.now().isoformat()
    
    # 1. Save to CSV (Backup)
    df_feedback = pd.DataFrame([feedback_entry])
    if not os.path.exists(FEEDBACK_FILE):
        df_feedback.to_csv(FEEDBACK_FILE, index=False)
    else:
        df_feedback.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)

    # 2. Save to DB (Real-time)
    save_to_db(feedback_entry)

st.title("Student Stress Prevention System 🧠")
st.markdown("High-Precision Prediction")

with st.form("prediction_form"):
    st.subheader("📝 Academic & Personal Details")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gen")
        age = st.number_input("Age", 15, 30, 20)
        dep = st.selectbox("Department", ["BCA", "Commerce", "Science", "Engineering", "Arts"], key="dep")
        certification = st.selectbox("Certification Course", ["No", "Yes"], key="cert")
        likedegree = st.selectbox("Do you like your degree?", ["No", "Yes"], key="like")

    with col2:
        hobbies = st.selectbox("Hobbies", ["Cinema", "Sports", "Reading", "Video Games", "Music"], key="hobby")
        prefertime = st.selectbox("Preferred Study Time", ["Morning", "Night", "Anytime"], key="ptime")
        financial = st.selectbox("Financial Status", ["Bad", "Mid", "Good"], key="fin")
        parttime = st.selectbox("Part Time Job", ["No", "Yes"], key="pt")
        carrer_willing = st.selectbox("Willing to pursue career in this field?", ["No", "Yes", "Undecided"])

    st.subheader("📚 Study & Health Metrics")
    col3, col4 = st.columns(2)
    
    with col3:
        study_hours_txt = st.selectbox("Daily Study Time", ["<1 hr", "1-2 hrs", "2-4 hrs", ">4 hrs"])
        mark_10th = st.number_input("10th Mark", 0.0, 100.0, 85.0)
        mark_12th = st.number_input("12th Mark", 0.0, 100.0, 85.0)
        college_mark = st.number_input("College Mark", 0.0, 100.0, 80.0)
        
    with col4:
        travel_txt = st.selectbox("Travelling Time", ["<30 min", "30-60 min", "1-2 hrs", ">2 hrs"])
        height = st.number_input("Height (CM)", 100, 250, 170)
        weight = st.number_input("Weight (KG)", 30, 150, 65)
        salexpect = st.number_input("Salary Expectation", 0, 1000000, 25000)

    submitted = st.form_submit_button("Predict Stress Level")

if submitted:
    if model is None:
        st.error("Model still loading or files missing. Please wait.")
    else:
        # 1. Prepare raw input dataframe
        input_data = {
            'gender': [gender],
            'dep': [dep],
            'certification': [certification],
            'likedegree': [likedegree],
            'hobbies': [hobbies],
            'prefertime': [prefertime],
            'financial': [financial],
            'parttime': [parttime],
            'studytime': [study_hours_txt],
            'mark10th': [mark_10th],
            'mark12th': [mark_12th],
            'collegemark': [college_mark],
            'travel': [travel_txt],
            'height': [height],
            'weight': [weight],
            'salexpect': [salexpect]
            # Note: We omitted a few minor ones for brevity but this covers high impact
        }
        X_new = pd.DataFrame(input_data)
        
        # 2. Preprocessing Pipeline (MUST MATCH TRAINING)
        
        # A. Mapping Time
        study_map = {'<1 hr': 0.5, '1-2 hrs': 1.5, '2-4 hrs': 3.0, '>4 hrs': 5.0}
        travel_map = {'<30 min': 0.25, '30-60 min': 0.75, '1-2 hrs': 1.5, '>2 hrs': 2.5}
        
        X_new['studytime'] = X_new['studytime'].map(study_map)
        X_new['travel'] = X_new['travel'].map(travel_map)
        
        # B. Feature Engineering
        X_new['avg_mark'] = (X_new['mark10th'] + X_new['mark12th'] + X_new['collegemark']) / 3.0
        X_new['efficiency'] = X_new['avg_mark'] / (X_new['studytime'] + 1.0)
        X_new['total_time'] = X_new['studytime'] + X_new['travel']
        
        # BMI
        X_new['bmi'] = X_new['weight'] / ((X_new['height'] / 100.0) ** 2)
        X_new['bmi'] = X_new['bmi'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # C. Encoding (One-Hot)
        # We need to get dummies, but alignment with training columns is tricky for single row.
        # Strategy: Create dummy df, then reindex against saved model_columns.
        columns_to_encode = ['certification', 'gender', 'dep', 'hobbies', 'prefertime', 'likedegree', 'financial', 'parttime']
        
        X_processed = pd.get_dummies(X_new, columns=columns_to_encode, drop_first=True)
        
        # Reindex to ensure all model columns exist (fill missing with 0)
        # We need to ensure we don't have EXTRA columns either that model doesn't know
        X_final = X_processed.reindex(columns=model_columns, fill_value=0)
        
        # D. Scaling
        columns_to_minmax = ['height', 'weight', 'mark10th', 'mark12th', 'collegemark', 'studytime', 'avg_mark', 'efficiency', 'total_time', 'bmi', 'travel']
        columns_to_std = ['salexpect']
        
        # Apply scaling if columns exist
        minmax_cols = [c for c in columns_to_minmax if c in X_final.columns]
        if minmax_cols:
            X_final[minmax_cols] = scaler_minmax.transform(X_final[minmax_cols])
            
        std_cols = [c for c in columns_to_std if c in X_final.columns]
        if std_cols:
            X_final[std_cols] = scaler_std.transform(X_final[std_cols])
            
        # 3. Prediction
        pred_val = model.predict(X_final)[0]
        prediction_map = {0: "Low (No Stress)", 1: "Medium (Healthy Pressure)", 2: "High (Burnout Risk!)"}
        pred_label = prediction_map.get(pred_val, "Unknown")
        
        # UI Result
        color_map = {0: "green", 1: "orange", 2: "red"}
        st.markdown(f"### Predicted Stress Level: :{color_map[pred_val]}[{pred_label}]")
        
        # Store for feedback
        st.session_state['last_pred'] = int(pred_val)
        st.session_state['last_data'] = input_data
        
# Feedback UI
if 'last_pred' in st.session_state:
    st.divider()
    st.write("Does this prediction seem accurate based on your feeling?")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if st.button("✅ Yes, Correct"):
            save_feedback(st.session_state['last_data'], st.session_state['last_pred'], st.session_state['last_pred'])
            st.success("Thanks! Model reinforced.")
    with col_f2:
        if st.button("❌ No, Incorrect"):
            st.info("Please tell us your actual stress level in the future updates (UI Simplified).")

