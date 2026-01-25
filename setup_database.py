
import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse
import os

# --- Configuration ---
# REPLACE THESE WITH YOUR ACTUAL CREDENTIALS
DB_USER = "root"
DB_PASSWORD = "Tomkar@13"  # Treat special characters with care if manual, script handles encoding.
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "student_stress_db"
TABLE_NAME = "feedback_data"
CSV_FILE = "human_feedback.csv"

def setup_and_load():
    # 1. URL Encode password to handle special characters like '@'
    encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
    
    # 2. Connect to MySQL Server (No Database selected yet)
    # Format: mysql+mysqlconnector://user:encoded_password@host:port
    server_conn_str = f"mysql+mysqlconnector://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}"
    
    try:
        print("Connecting to MySQL Server...")
        server_engine = create_engine(server_conn_str)
        
        # Create Database if not exists
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
            print(f"Database '{DB_NAME}' created or already exists.")
            
    except Exception as e:
        print(f"Error connecting to MySQL Server: {e}")
        return

    # 3. Connect to the specific Database
    db_conn_str = f"{server_conn_str}/{DB_NAME}"
    try:
        db_engine = create_engine(db_conn_str)
        
        # 4. Define Table Schema (Optional, pandas can infer, but explicit is safer)
        # Using simple SQL for clarity, or letting pandas do it.
        # Let's let pandas create it since we have the CSV structure.
        
        if os.path.exists(CSV_FILE):
            print(f"Reading {CSV_FILE}...")
            df = pd.read_csv(CSV_FILE)
            
            # Write to SQL
            print(f"Writing data to table '{TABLE_NAME}'...")
            df.to_sql(name=TABLE_NAME, con=db_engine, if_exists='append', index=False)
            print("Data loaded successfully!")
            
            # Verify
            with db_engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
                count = result.scalar()
                print(f"Total rows in '{TABLE_NAME}': {count}")
        else:
            print(f"Warning: {CSV_FILE} not found. Database created but table is empty.")
            
            # Create empty table manually if CSV missing, just to satisfy request
            with db_engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        studytime INT,
                        mark10th INT,
                        gender VARCHAR(50),
                        Predicted_Stress INT,
                        Actual_Stress INT,
                        Timestamp VARCHAR(100)
                    )
                """))
                print(f"Empty table '{TABLE_NAME}' ensured.")

    except Exception as e:
        print(f"Error checking/writing to database: {e}")

if __name__ == "__main__":
    setup_and_load()
