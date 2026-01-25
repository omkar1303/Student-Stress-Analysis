
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
import urllib.parse

DB_USER = "root"
DB_PASSWORD = "Tomkar@13"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "student_stress_db"
TABLE_NAME = "feedback_data"

def inspect_table():
    encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
    conn_str = f"mysql+mysqlconnector://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(conn_str)
    
    inspector = inspect(engine)
    columns = inspector.get_columns(TABLE_NAME)
    print(f"Columns in '{TABLE_NAME}':")
    for col in columns:
        print(f"- {col['name']} ({col['type']})")

if __name__ == "__main__":
    inspect_table()
