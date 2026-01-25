import pandas as pd
from sqlalchemy import create_engine, types
import os

# --- Configuration ---
# Update these with your actual MySQL credentials
DB_CONFIG = {
    "user": "root",
    "password": "Tomkar@13",
    "host": "localhost",
    "port": "3306",
    "database": "student_system"
}

CSV_FILE = "human_feedback.csv"
TABLE_NAME = "feedback_data"

def load_data_to_mysql():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print("Reading CSV data...")
    df = pd.read_csv(CSV_FILE)
    print(f"Read {len(df)} rows from {CSV_FILE}.")

    # Create SQL Alchemy Engine
    # Format: mysql+mysqlconnector://user:password@host:port/database
    conn_str = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    
    try:
        print("Connecting to database...")
        # Note: If the database does not exist, this might fail unless created beforehand.
        # For simplicity, we assume the DB exists or connection string targets an existing one.
        engine = create_engine(conn_str)
        
        # Define datatypes if needed, or let pandas infer
        # 'if_exists' options: 'fail', 'replace', 'append'
        print(f"Writing to table '{TABLE_NAME}'...")
        df.to_sql(name=TABLE_NAME, con=engine, if_exists='append', index=False)
        
        print("Successfully loaded data into MySQL!")
        
    except Exception as e:
        print(f"\nError: Could not connect or write to Database.\nDetails: {e}")
        print("\nTroubleshooting:")
        print("1. Check if MySQL server is running.")
        print("2. Verify credentials in the script.")
        print("3. Ensure the database name exists.")

if __name__ == "__main__":
    load_data_to_mysql()
