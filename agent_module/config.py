import os
from dotenv import load_dotenv
load_dotenv()
API_KEY=os.getenv("GROQ_API_KEY", "")
OPENROUTER=os.getenv("OPENROUTER", "")
DATA_DB_URI=os.getenv("DATA_DB_URI", "postgresql://postgres:fx818@localhost:5432/project_automation_local")
MEMORY_DB_URI=os.getenv("MEMORY_DB_URI", "sqlite:///../database/user_memory.sqlite3")
MEMORY_DB_PATH=os.getenv("MEMORY_DB_PATH", "../database/user_memory.sqlite3")

import psycopg2
import pandas as pd

def get_table_relationships(uri: str):
    query = """
    SELECT
        tc.table_schema,
        tc.table_name AS foreign_table,
        kcu.column_name AS foreign_column,
        ccu.table_name AS primary_table,
        ccu.column_name AS primary_column
    FROM
        information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
        ON ccu.constraint_name = tc.constraint_name
        AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
    ORDER BY foreign_table, foreign_column;
    """
    
    conn = psycopg2.connect(uri)
    df = pd.read_sql(query, conn)
    conn.close()
    data = df.to_dict(orient="records")
    return data

# data = get_table_relationships(DATA_DB_URI)
# print(data)
# for d in data:
    # print(f"Foreign Table: {d['foreign_table']}, Foreign Column: {d['foreign_column']}, Primary Table: {d['primary_table']}, Primary Column: {d['primary_column']}")