import json
import sqlite3
from typing import Dict, Any, Optional
import os
import sys
from sqlalchemy import create_engine, text, MetaData, Table, Column, String

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import agent_module.config as config
# This is about the db from where we would get data about which user will ask query
def get_table_info_str(query_db_object):
    """Get all table information from the database as a string"""
    table_infos = []
    try:
        # Get all table names from sqlite_master
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        result = query_db_object.run(query)
        
        # Handle different result formats
        tables = []
        if isinstance(result, str):
            # If result is a string, try to parse it
            import ast
            try:
                parsed_result = ast.literal_eval(result)
                if isinstance(parsed_result, list):
                    tables = [row[0] if isinstance(row, tuple) else row for row in parsed_result]
            except:
                # If parsing fails, try splitting the string
                lines = result.strip().split('\n')
                tables = [line.strip() for line in lines if line.strip()]
        elif isinstance(result, list):
            # If result is a list, extract table names
            for row in result:
                if isinstance(row, tuple) and len(row) > 0:
                    tables.append(row[0])
                elif isinstance(row, dict) and 'name' in row:
                    tables.append(row['name'])
                elif isinstance(row, str):
                    tables.append(row)
        
        for table in tables:
            if not table or not isinstance(table, str):
                continue
                
            try:
                columns_info_result = query_db_object.run(f"PRAGMA table_info({table});")
                columns_with_types = []
                columns_only = []
                
                # Parse column info
                columns_info = []
                if isinstance(columns_info_result, str):
                    # Try to parse string result
                    try:
                        import ast
                        columns_info = ast.literal_eval(columns_info_result)
                    except:
                        continue
                elif isinstance(columns_info_result, list):
                    columns_info = columns_info_result
                
                if columns_info and isinstance(columns_info, list) and len(columns_info) > 0:
                    for col in columns_info:
                        try:
                            if isinstance(col, tuple) and len(col) >= 3:
                                # col: (cid, name, type, notnull, dflt_value, pk)
                                columns_with_types.append(f"{col[1]} ({col[2]})")
                                columns_only.append(col[1])
                            elif isinstance(col, dict):
                                name = col.get('name', '')
                                col_type = col.get('type', '')
                                if name:
                                    columns_with_types.append(f"{name} ({col_type})")
                                    columns_only.append(name)
                        except (IndexError, TypeError):
                            continue
                
                if columns_only:  # Only add table if we found columns
                    table_info = f"Table '{table}':\n"
                    table_info += f"  Columns: {', '.join(str(col) for col in columns_only)}\n"
                    table_info += f"  Detailed: {', '.join(str(col) for col in columns_with_types)}"
                    table_infos.append(table_info)
                    
            except Exception as e:
                print(f"Error getting info for table {table}: {e}")
                continue
                
    except Exception as e:
        print(f"Error getting table info: {e}")
        # Fallback to basic table list
        return "No tables found in database"
        
    return "\n\n".join(table_infos) if table_infos else "No tables found in database"

def get_table_info_pg_str(query_db_object):
    """Get all table information from a PostgreSQL database as a string"""
    table_infos = []
    try:
        # Get all table names from information_schema
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        result = query_db_object.run(query)

        tables = []
        if isinstance(result, str):
            # Try parsing string output
            import ast
            try:
                parsed_result = ast.literal_eval(result)
                if isinstance(parsed_result, list):
                    tables = [row[0] if isinstance(row, (tuple, list)) else row for row in parsed_result]
            except:
                lines = result.strip().split('\n')
                tables = [line.strip() for line in lines if line.strip()]
        elif isinstance(result, list):
            for row in result:
                if isinstance(row, (tuple, list)) and len(row) > 0:
                    tables.append(row[0])
                elif isinstance(row, dict) and 'table_name' in row:
                    tables.append(row['table_name'])
                elif isinstance(row, str):
                    tables.append(row)

        for table in tables:
            if not table or not isinstance(table, str):
                continue

            try:
                # Get column info from information_schema
                columns_info_result = query_db_object.run(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position;
                """)

                columns_with_types = []
                columns_only = []

                columns_info = []
                if isinstance(columns_info_result, str):
                    try:
                        import ast
                        columns_info = ast.literal_eval(columns_info_result)
                    except:
                        continue
                elif isinstance(columns_info_result, list):
                    columns_info = columns_info_result

                if columns_info and isinstance(columns_info, list) and len(columns_info) > 0:
                    for col in columns_info:
                        try:
                            if isinstance(col, (tuple, list)) and len(col) >= 2:
                                name, dtype = col[0], col[1]
                                columns_with_types.append(f"{name} ({dtype})")
                                columns_only.append(name)
                            elif isinstance(col, dict):
                                name = col.get('column_name', '')
                                dtype = col.get('data_type', '')
                                if name:
                                    columns_with_types.append(f"{name} ({dtype})")
                                    columns_only.append(name)
                        except (IndexError, TypeError):
                            continue

                if columns_only:
                    table_info = f"Table '{table}':\n"
                    table_info += f"  Columns: {', '.join(columns_only)}\n"
                    table_info += f"  Detailed: {', '.join(columns_with_types)}"
                    table_infos.append(table_info)

            except Exception as e:
                print(f"Error getting info for table {table}: {e}")
                continue

    except Exception as e:
        print(f"Error getting table info: {e}")
        return "No tables found in database"

    return "\n\n".join(table_infos) if table_infos else "No tables found in database"


# Memory db path
memory_db_path = config.MEMORY_DB_PATH

def save_user_memory(username: str, memory: Dict, email: str):
    conn = sqlite3.connect(memory_db_path)
    try:
        # Connect to the SQLite database
        cursor = conn.cursor()
        # Convert the dictionary to a JSON string
        prev_mem = get_user_memory(username)
        if prev_mem.get("status"):
            memory = {**prev_mem.get("memory", {}), **memory}
        else:
            memory = memory
        memory_json = json.dumps(memory)
        # Use a parameterized query with placeholders (?, ?, ?)
        clear_user_memory(username)
        command = """
            INSERT into user_memory (username, memory, email) values(?, ?, ?)
        """
        
        # Pass the parameters as a tuple to the run method
        cursor.execute(command, (username, memory_json, email))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        print("Failed to save user memory due to the following error:", e)
        return False

def clear_user_memory(username: str):
    conn = sqlite3.connect(memory_db_path)
    cursor = conn.cursor()
    try:
        command = f"""
            DELETE FROM user_memory WHERE USERNAME = '{username}'
        """
        cursor.execute(command)
        conn.commit()
    except Exception as e:
        print("Failed to clear user memory due to the following error: ", e)
        return False
    finally:
        conn.close()
    return True

def get_user_memory(username: str) -> Dict:
    conn = sqlite3.connect(memory_db_path)
    cursor = conn.cursor()
    result = cursor.execute(f"SELECT memory, email FROM user_memory WHERE username = '{username}'").fetchall()
    print("The result from get_user_memory is ", result)
    if result:
        memory, email = result[0]
        memory = json.loads(memory) if memory else {}
        return {"memory": memory, "email": email, "status": True}
    return {"memory": "", "email": "", "status": False}

def create_db(db_name: str):
    conn = sqlite3.connect(db_name)
    conn.close()

def create_user_memory_table():
    command = """
        CREATE TABLE IF NOT EXISTS user_memory (
        username TEXT PRIMARY KEY,
        memory TEXT NOT NULL,
        email TEXT
    )"""
    conn = sqlite3.connect(memory_db_path)
    cursor = conn.cursor()
    cursor.execute(command)
    conn.commit()
    conn.close()
