
import pandas as pd
from sqlalchemy import create_engine
import os

# --- Configuration ---
DB_NAME = "student_stress.db"
TABLE_NAME = "feedback_data"
CSV_FILE = "human_feedback.csv"

def load_data_to_sqlite():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print("Reading CSV data...")
    df = pd.read_csv(CSV_FILE)
    
    # Rename columns to match the app's schema if necessary
    # App expects: distinct keys mapped. 
    # CSV has: studytime,mark10th,gender,Predicted_Stress,Actual_Stress,Timestamp
    # DB expects: snake_case usually if we follow the pattern I set in app.py
    
    # Let's check keys in app.py from previous turns:
    # "studytime", "mark10th", "gender", "predicted_stress", "actual_stress", "created_at"
    
    df.rename(columns={
        "Predicted_Stress": "predicted_stress",
        "Actual_Stress": "actual_stress",
        "Timestamp": "created_at"
    }, inplace=True)

    print(f"Read {len(df)} rows from {CSV_FILE}.")

    # SQLite Connection
    conn_str = f"sqlite:///{DB_NAME}"
    try:
        print(f"Connecting to SQLite ({DB_NAME})...")
        engine = create_engine(conn_str)
        
        print(f"Writing to table '{TABLE_NAME}'...")
        # if_exists='append' adds to existing, 'replace' would overwrite
        df.to_sql(name=TABLE_NAME, con=engine, if_exists='append', index=False)
        
        print("Successfully loaded data into SQLite!")
        
        # Verify
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
            print(f"Total rows in database: {result.scalar()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    load_data_to_sqlite()
