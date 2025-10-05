from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import os, sys, json, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_module.sql_agent import SQLAgent, State
from database.dbconfig import get_table_info_pg_str, get_table_info_str, clear_user_memory, get_user_memory, save_user_memory
from langchain_community.utilities import SQLDatabase
import agent_module.config as config 
load_dotenv()

app = FastAPI()
secret_key = os.getenv("SESSION_SECRET_KEY", "default_secret_key")
app.add_middleware(SessionMiddleware, secret_key=secret_key)


# ---------------- Helpers ----------------
def sanitize_for_thread_id(s: str) -> str:
    """Make a safe thread id from an email (alphanumeric + underscores)."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", s or "anonymous")


def get_initial_state_for_user(username: str, user_query: str | None = None) -> State:
    """Return a fresh State for a user's LangGraph invocation."""
    return State(
        username=username,
        user_query=user_query,
        resolved_user_query=None,
        sql_query=None,
        sql_query_response=None,
        response=None,
        error=None,
        context=None
    )

def get_user_thread_config(username: str) -> dict:
    """Each user gets their own thread_id for LangGraph state."""
    tid = f"certificate-flow-{sanitize_for_thread_id(username)}"
    return {"configurable": {"thread_id": tid}}


static_files_dir = os.path.join(os.path.dirname(__file__), "static")

# ---------------- Routes ----------------
@app.get("/")
def read_root():
    return {"status": "API is running well and good."}

@app.get("/home")
def home():
    """Serve the index.html file as the root page."""
    return FileResponse(os.path.join(static_files_dir, 'index.html'))

@app.post("/query")
def handle_query(request: Request):
    """Handle a user query."""
    username= request.session.get("username", "anonymous")
    user_query = request.session.get("user_query", None)
    if not username:
        return {"error": "No username found in session."}
    if not user_query:
        return {"error": "No user query found in session."}
    print(f"User '{username}' query: {user_query}")
    state = get_initial_state_for_user(username, user_query)
    thread_config = get_user_thread_config(username)
    agent = SQLAgent(config.DATA_DB_URI, config.MEMORY_DB_URI)
    agent.graph.invoke(state, thread_config=thread_config)
    return {"response": state.get("response", ""), "error": state.get("error", "")}

@app.post("/testing_query")
def test_handle_query(username: str, user_query: str):
    """Handle a user query."""
    if not username:
        return {"error": "No username found in session."}
    if not user_query:
        return {"error": "No user query found in session."}
    print(f"User '{username}' query: {user_query}")
    state = get_initial_state_for_user(username, user_query)
    thread_config = get_user_thread_config(username)
    agent = SQLAgent(config.DATA_DB_URI, config.MEMORY_DB_URI)
    state = agent.graph.invoke(state, thread_config=thread_config)
    return {"response": state.get("response", ""), "error": state.get("error", ""),"sql_query_response": state.get("sql_query_response", "")}


@app.delete("/clear_memory")
def clear_memory(username: str):
    """Clear user memory."""
    if not username:
        return {"error": "No username provided."}
    success = clear_user_memory(username)
    if success:
        return {"status": f"Memory cleared for user '{username}'."}
    else:
        return {"error": "Failed to clear memory."}
    
@app.post("/get_memory")
def get_memory(username: str):
    """Get user memory."""
    if not username:
        return {"error": "No username provided."}
    memory = get_user_memory(username)
    if memory.get("status"):
        return {"memory": memory.get("memory", {})}
    else:
        return {"error": "No memory found for user."}
    
@app.post("/get_db_schema")
def get_db_schema():
    """Get database schema."""
    db = SQLDatabase.from_uri(config.DATA_DB_URI)
    schema = get_table_info_pg_str(db)
    return {"schema": schema}