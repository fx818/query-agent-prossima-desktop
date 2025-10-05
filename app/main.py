
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent_module.sql_agent import SQLAgent, State
from database.dbconfig import get_table_info_pg_str, get_table_info_str
import json
from langchain_community.utilities import SQLDatabase

data_db_uri = "postgresql://postgres:fx818@localhost:5432/project_automation_local"
memory_db_uri = "sqlite:///../database/user_memory.sqlite3"


# SQLDatabase.from_uri(data_db_uri)
# mem_obj = SQLDatabase.from_uri(memory_db_uri)


agent = SQLAgent(data_db_uri, memory_db_uri)
initial_state = State(username="anurag", user_query="Give me details of suppliers whose name starts with prag", resolved_user_query=None, sql_query=None, sql_query_response=None, response=None, error=None, context=None)

agent.graph.invoke(initial_state)
# print(state)
# for s in agent.graph.stream(initial_state):
    # state = s  # This will keep updating until the last state
# print(state)