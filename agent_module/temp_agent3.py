import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import re
import sqlite3
from typing import Dict, Any, Optional
# CHANGED: switched to OpenAI client for OpenRouter
from openai import OpenAI  
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, NotRequired
from langchain_core.utils import convert_to_secret_str
import agent_module.config as config
from database.dbconfig import get_table_info_str, get_table_info_pg_str , get_user_memory, save_user_memory, clear_user_memory

class State(TypedDict):
    """State representation for sql agent."""
    username: str
    user_query: Optional[str]
    resolved_user_query: Optional[str]
    sql_query: Optional[str]
    sql_query_response: Optional[str]
    response: Optional[str]
    error: Optional[str]
    context: Optional[str]
    tables: Optional[list]

print("The api is ", config.API_KEY)

class SQLAgent:
    def __init__(self, data_db_uri: str, user_memory_db_uri: str):
        self.data_db_uri = data_db_uri
        self.user_memory_db_uri = user_memory_db_uri
        
        # CHANGED: use OpenRouter client directly
        self.llm = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.API_KEY
        )

        self.db = SQLDatabase.from_uri(data_db_uri)
        # User memory db object
        self.user_memory_db_object = SQLDatabase.from_uri(user_memory_db_uri)
        # Build the graph
        self.graph = self.build_graph()

    def add_memory_context(self, state: State):
        username = state.get("username", "")
        print("username is ", username)
        context = get_user_memory(username)
        if not context.get("status"):
            print("No prev context found")
        else:
            state["context"] = context.get("memory")
            print("Added the memory context")
        return state
    
    def resolve_user_query(self, state: State):
        system_message = """
            You are a Query Resolver Assistant. Here is the database dialect {dialect}
            Your role is to process user queries that are meant to fetch information from a database.
            Here is the database schema: 
            {schema}
            Follow the schema properly, do not deviate from it
            Your task is to convert informal, incomplete, or grammatically incorrect user queries into clear, correct questions. It should be in the form of natural language that can be easily understood not SQL query
            Here is the user query: {user_que} along with the prev context {context}
            Follow these rules carefully:
            Understand the User Query
            Take the raw input from the user.
            The input may contain incomplete, grammatically incorrect, or informal language.
            Ensure the resolved query keeps the exact intent of the user.
            Do not add extra conditions or assumptions unless necessary to make the query correct.
            Output Format
            Always return the result in strict JSON format with the following structure:
            {{
            "user_que": "<original user query>",
            "resolved_ques": "<cleaned, corrected, and database-ready query>"
            }}
            Additional Rules
            Do not provide explanations, comments, or extra text outside of the JSON.
            Preserve important keywords, entities, or parameters mentioned by the user.
            If the query is ambiguous, resolve it into the most reasonable database-compatible form without losing meaning.
            Example
            User Input:
            give me detail of emp salry dept id 5
            LLM Output:
            {{
            "user_que": "give me detail of emp salry dept id 5",
            "resolved_ques": "Get details of employee salary where department_id = 5"
            }}           
        """
        user_que = state.get("user_query", "")
        if not user_que:
            print("No question from user found")
            return
        query_prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", user_que)]
        )
        messages = query_prompt_template.format_messages(
            dialect = self.db.dialect,
            schema = get_table_info_pg_str(self.db),
            user_que=user_que,
            context=state.get("context", "")
        )

        # CHANGED: OpenRouter call
        completion = self.llm.chat.completions.create(
            model="meta-llama/llama-4-scout",
            messages=[{"role": m.type, "content": m.content} for m in messages],
        )
        response_text = completion.choices[0].message["content"].strip()

        response_json = {}
        resolved_ques = ""

        try:
            if isinstance(response_text, str):
                response_json = json.loads(response_text)
            elif isinstance(response_text, dict):
                response_json = response_text
            else:
                raise ValueError("Unexpected response type")

        except Exception:
            match = re.search(r'\{.*\}', str(response_text), re.DOTALL)
            if match:
                try:
                    response_json = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print("⚠️ Regex found JSON but it was invalid.")
            else:
                print("⚠️ No JSON found in response.")

        if isinstance(response_json, dict):
            resolved_ques = response_json.get("resolved_ques", "")
        if not resolved_ques:
            print("Dude, ques could not be resolved")
            state["resolved_user_query"] = user_que
        else:
            state["resolved_user_query"] = resolved_ques

        print("the resolved ques is ", resolved_ques)
        return state

    def identify_tables(self, state: State):
        user_que = state.get("user_query", "")
        if not user_que:
            print("No question from user found")
            return
        system_message = """ ... (unchanged long prompt) ... """

        query_prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", user_que)]
        )
        messages = query_prompt_template.format_messages(
            dialect = self.db.dialect,
            schema = get_table_info_pg_str(self.db),
            user_que=user_que
        )

        # CHANGED: OpenRouter call
        completion = self.llm.chat.completions.create(
            model="meta-llama/llama-4-scout",
            messages=[{"role": m.type, "content": m.content} for m in messages],
        )
        response_text = completion.choices[0].message["content"].strip()

        response_json = {}
        tables = []
        try:
            if isinstance(response_text, str):
                response_json = json.loads(response_text)
            elif isinstance(response_text, dict):
                response_json = response_text
        except Exception:
            match = re.search(r'\{.*\}', str(response_text), re.DOTALL)
            if match:
                try:
                    response_json = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print("⚠️ Regex found JSON but it was invalid.")
            else:
                print("⚠️ No JSON found in response.")

        if isinstance(response_json, dict):
            tables = response_json.get("tables", [])
        if not tables:
            print("No relevant tables found")
            state["tables"] = []
        else:         
            state["tables"] = tables
        tables = list(set(tables))

        # rest of your logic unchanged
        # ...

        return state

    def write_query(self, state: State):
        system_message = """ ... (unchanged long SQL generation prompt) ... """
        user_que = state.get("user_query", "")
        user_resolved_que = state.get("resolved_user_query", "")
        dialect = self.db.dialect
        schema = get_table_info_pg_str(self.db)

        query_prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", user_que)]
        )
        messages = query_prompt_template.format_messages(
            user_que=user_que,
            user_resolved_question=user_resolved_que,
            dialect=dialect,
            schema=schema,
            memory_context=state.get("context", "No relevant memory found"),
            table_data=state.get("tables")
        )

        # CHANGED: OpenRouter call
        completion = self.llm.chat.completions.create(
            model="meta-llama/llama-4-scout",
            messages=[{"role": m.type, "content": m.content} for m in messages],
        )
        response_text = completion.choices[0].message["content"].strip()

        match = re.search(r"```sql\s+(.*?)```", response_text, re.DOTALL)
        sql_query = match.group(1).strip() if match else response_text.strip()

        sql_query = sql_query.replace("\n", " ")
        state["sql_query"] = sql_query

        print("The generated query is ", sql_query)
        return state

    def execute_query(self, state: State):
        sql_query = state.get("sql_query", "")
        if not sql_query:
            print("No SQL query found to execute")
            state["error"] = "No SQL query to execute"
            return
        try:
            tool = QuerySQLDatabaseTool(db=self.db)
            result = tool.invoke(sql_query)
            state["sql_query_response"] = result
        except Exception as e:
            print(f"Error executing SQL: {e}")
            state["error"] = str(e)
            state["sql_query_response"] = ""
        print("the sql query response is ", state["sql_query_response"])
        return state

    def fix_query(self, state: State):
        # CHANGED: replace invoke with OpenRouter
        error_msg = state.get("error", "")
        failed_query = state.get("sql_query", "")
        user_query = state.get("user_query", "")
        schema = get_table_info_pg_str(self.db)      
        table_data = state.get("tables", "")

        if not error_msg:
            return state

        prompt = f""" ... (your unchanged fix query prompt) ... """

        completion = self.llm.chat.completions.create(
            model="meta-llama/llama-4-scout",
            messages=[{"role": "system", "content": prompt}],
        )
        response_text = completion.choices[0].message["content"].strip()

        match = re.search(r"```sql\s+(.*?)```", response_text, re.DOTALL)
        fixed_query = match.group(1).strip() if match else response_text.strip()

        state["sql_query"] = fixed_query
        state["error"] = None
        print("[FixQuery] New SQL:", fixed_query)
        return state

    def generate_answer(self, state: State):
        system_message = """ ... (unchanged answer prompt) ... """
        user_que = state.get("user_query", "")
        sql_query_response = state.get("sql_query_response", "")
        context = state.get("context", "")

        query_prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", user_que)]
        )
        messages = query_prompt_template.format_messages(
            query_result=sql_query_response,
            context=context
        )

        # CHANGED: OpenRouter call
        completion = self.llm.chat.completions.create(
            model="meta-llama/llama-4-scout",
            messages=[{"role": m.type, "content": m.content} for m in messages],
        )
        response_text = completion.choices[0].message["content"].strip()

        if not response_text: 
            print("Response could not be generated or parsed")
            return state
        state["response"] = response_text
        print("the generated response is ", state["response"])
        memory = {state.get("user_query", ""): state.get("response", "")}
        save_user_memory(state.get("username", ""), memory, "email")
        return state

    # build_graph unchanged
    def build_graph(self):
        builder = StateGraph(State)
        # ... (same edges as before)
        return builder.compile()

# Execution code unchanged
data_db_uri = config.DATA_DB_URI
memory_db_uri = config.MEMORY_DB_URI

agent = SQLAgent(data_db_uri, memory_db_uri)
initial_state = State(username="818", user_query="Show me the name of all items whose all document has been been completed that is they are not pending.", resolved_user_query=None, sql_query=None, sql_query_response=None, response=None, error=None, context=None, tables=None)
final_state = agent.graph.invoke(initial_state)

print("the extracted tables are ", final_state.get("tables", []))
print("The sql response is ", final_state.get("sql_query_response", ""))
print("The response is ", final_state.get("response", ""))
