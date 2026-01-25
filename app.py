
import streamlit as st
import pandas as pd
import xgboost as xgb
import os
from datetime import datetime

# Load Model
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    try:
        model.load_model("student_stress_model.json")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return model

model = load_model()

import urllib.parse
from sqlalchemy import create_engine, text

# --- Configuration (SQLite) ---
DB_NAME = "student_stress.db"
TABLE_NAME = "feedback_data"

# Feedback File (Keep CSV as backup)
FEEDBACK_FILE = "human_feedback.csv"

def save_to_db(data_dict):
    try:
        # SQLite Connection String
        conn_str = f"sqlite:///{DB_NAME}"
        engine = create_engine(conn_str)
        
        # Map keys to match Database Schema
        db_data = {
            "studytime": data_dict.get("studytime"),
            "mark10th": data_dict.get("mark10th"),
            "gender": data_dict.get("gender"),
            "predicted_stress": data_dict.get("Predicted_Stress"),
            "actual_stress": data_dict.get("Actual_Stress"),
            "created_at": data_dict.get("Timestamp")
        }
        
        df = pd.DataFrame([db_data])
        df.to_sql(name=TABLE_NAME, con=engine, if_exists='append', index=False)
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
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
st.markdown("Enter student details to predict stress levels. **Your feedback helps improve the model!**")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        study_hours = st.slider("Daily Study Time (Hours)", 0, 15, 5, key="study")
        sleep_hours = st.slider("Sleep Quality (1-5)", 1, 5, 3, key="sleep") 
        mark_10th = st.number_input("10th Mark", 0, 100, 75, key="m10")
        mark_12th = st.number_input("12th Mark", 0, 100, 75, key="m12")
        college_mark = st.number_input("College Mark", 0, 100, 75, key="mcol")
        
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gen")
        financial = st.selectbox("Financial Status", ["Bad", "Mid", "Good"], key="fin")
        part_time = st.selectbox("Part Time Job", ["Yes", "No"], key="pt")
        hobbies = st.selectbox("Hobbies", ["Cinema", "Sports", "Reading", "Video Games"], key="hobby")

    submitted = st.form_submit_button("Predict Stress Level")

if submitted:
    # --- Prediction Mock Logic (Replace with real preprocessing/prediction) ---
    # Ideally, you load the scaler/encoder from training and transform these inputs.
    # For now, we simulate a prediction to demonstrate the feedback loop UI.
    
    # Simple heuristic for demo
    score = (study_hours * 2) + (6 - sleep_hours) * 2
    if score > 20: pred_val = 2 # High
    elif score > 10: pred_val = 1 # Medium
    else: pred_val = 0 # Low
    
    prediction_map = {0: "Low", 1: "Medium", 2: "High"}
    pred_label = prediction_map[pred_val]
    
    # Store session state for feedback
    st.session_state['last_pred'] = pred_val
    st.session_state['last_data'] = {
        "studytime": study_hours,
        "mark10th": mark_10th,
        "gender": gender,
        # ... add others
    }
    
    st.success(f"Predicted Stress Level: **{pred_label}**")

# Feedback UI (Outside form to persist)
if 'last_pred' in st.session_state:
    st.divider()
    st.subheader("Human Feedback Loop")
    st.write("Was this prediction correct?")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if st.button("✅ Yes, Correct"):
            save_feedback(st.session_state['last_data'], st.session_state['last_pred'], st.session_state['last_pred'])
            st.success("Feedback Saved! This data will be used to reinforce the model.")
            
    with col_f2:
        actual_label = st.selectbox("No, it should be:", ["Low", "Medium", "High"])
        if st.button("💾 Submit Correction"):
            rev_map = {"Low": 0, "Medium": 1, "High": 2}
            save_feedback(st.session_state['last_data'], st.session_state['last_pred'], rev_map[actual_label])
            st.warning("Correction Saved! The model will learn from this error.")

st.markdown("---")
st.subheader("📊 Live Feedback Data")
try:
    # Show data directly from DB to prove it works
    conn_str = f"sqlite:///{DB_NAME}"
    engine = create_engine(conn_str)
    with engine.connect() as conn:
        df_view = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY created_at DESC", conn)
        st.dataframe(df_view)
        
        # Download Button
        with open(DB_NAME, "rb") as file:
            btn = st.download_button(
                label="📥 Download Database File",
                data=file,
                file_name="student_stress.db",
                mime="application/x-sqlite3"
            )
except Exception as e:
    st.error(f"Could not load data: {e}")

if os.path.exists(FEEDBACK_FILE):
    # Optional: Keep showing CSV count or remove if confusing
    pass
