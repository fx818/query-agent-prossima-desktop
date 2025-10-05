import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import re
import sqlite3
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
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
        self.llm = ChatOpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=convert_to_secret_str(config.API_KEY),
                    model="llama-3.1-8b-instant",
                    temperature=0,
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
        raw_response = self.llm.invoke(messages)
        # Extract text (handles both string and message object cases)
        response_text = getattr(raw_response, "content", None) or str(raw_response)

        response_json = {}
        resolved_ques = ""

        try:
            # First try direct JSON parse
            if isinstance(response_text, str):
                response_json = json.loads(response_text)
            elif isinstance(response_text, dict):
                response_json = response_text
            else:
                raise ValueError("Unexpected response type")

        except Exception:
            # Fallback: extract JSON substring using regex
            match = re.search(r'\{.*\}', str(response_text), re.DOTALL)
            if match:
                try:
                    response_json = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print("⚠️ Regex found JSON but it was invalid.")
            else:
                print("⚠️ No JSON found in response.")

        # Safely extract resolved query
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
        system_message = """
            You are a STRICT Table Identifier Assistant. 
            You will be given:
            - The database dialect: {dialect}
            - The complete database schema: {schema}
            - A natural language user query: {user_que}

            YOUR JOB:
            Identify ALL tables from the schema that are directly or indirectly relevant to answering the user query. 
            This means:
            1. Include tables explicitly mentioned in the query.
            2. Include tables that are obviously needed to resolve the query (e.g., foreign keys, relationships).
            3. Include tables that might reasonably be related to the query, even if not explicitly named, as long as they could help in answering it.
            4. NEVER miss a possible table. When in doubt, include it.

            CRITICAL RULES:
            - The output MUST be in STRICT JSON only.
            - NO explanations, comments, or natural language text outside the JSON.
            - Use ONLY table names exactly as they appear in the schema. Case-sensitive.
            - If truly no relevant tables exist (rare), return an empty list.

            OUTPUT FORMAT (mandatory):
            {{
            "user_que": "<original user query>",
            "tables": ["<table1>", "<table2>", ...]
            }}

            EXAMPLE:
            User Input:
            give me detail of emp salry dept id 5

            LLM Output:
            {{
            "user_que": "give me detail of emp salry dept id 5",
            "tables": ["employee", "salary", "department"]
            }}

            FAILURE CONDITIONS (never do these):
            - Do NOT explain your reasoning.
            - Do NOT add extra text, notes, or comments.
            - Do NOT skip related tables.
            - Do NOT change table names.

            Think aggressively. If the user asks about something, grab every table that could possibly answer that query. Better to include extra tables than to miss one.
        """

        query_prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", user_que)]
        )
        messages = query_prompt_template.format_messages(
            dialect = self.db.dialect,
            schema = get_table_info_pg_str(self.db),
            user_que=user_que
        )
        raw_response = self.llm.invoke(messages)
        # Extract text (handles both string and message object cases)
        response_text = getattr(raw_response, "content", None) or str(raw_response)
        response_json = {}
        tables = []
        try:
            # First try direct JSON parse
            if isinstance(response_text, str):
                response_json = json.loads(response_text)
            elif isinstance(response_text, dict):
                response_json = response_text
            else:
                raise ValueError("Unexpected response type")

        except Exception:
            # Fallback: extract JSON substring using regex
            match = re.search(r'\{.*\}', str(response_text), re.DOTALL)
            if match:
                try:
                    response_json = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print("⚠️ Regex found JSON but it was invalid.")
            else:
                print("⚠️ No JSON found in response.")
        # Safely extract tables
        if isinstance(response_json, dict):
            tables = response_json.get("tables", [])
        if not tables:
            print("No relevant tables found")
            state["tables"] = []
        else:         
            state["tables"] = tables
        tables = list(set(tables))  # Deduplicate
        # get all the data from these tables
        table_data = []
        for table in tables:
            try:
                query = f"SELECT * FROM {table} LIMIT 5;"
                result = self.db.run(query)
                col_query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = '{table}';
                """
                col_names = str(self.db.run(col_query))
                table_data.append({table: col_names + str(result)})

            except Exception as e:
                print(f"Error querying table '{table}': {e}")
        print("the identified tables are ", tables)
        print("########################################################################################################")
        print("########################################################################################################")
        print()
        print()
        print()
        
        print("the table data is ", table_data)
        
        print("########################################################################################################")
        print("########################################################################################################")
        print("########################################################################################################")
        print()
        print()
        print()
        
        system_prompt = """
            You are an Expert SQL Assistant whose ONLY job is to convert structured database information into 
            a clear and natural human-readable answer. 

            You will be given:
            - A user query: {user_que}
            - The database dialect: {dialect}
            - The names of relevant tables: {tables}
            - The schema (columns, datatypes, relationships) of those tables: {schema}
            - The actual data rows from those tables: {table_data}

            Your task:
            1. Carefully read the user query and understand the intent.
            2. Look only at the provided table_data (NOT the schema or dialect for final answer).
            3. Based strictly on the table_data, generate a natural language answer to the user query.
            - Be concise but complete.
            - If multiple rows are relevant, summarize them in a human-friendly way 
                (lists, counts, comparisons, etc.).
            - Do NOT output SQL queries or technical details.
            - Do NOT invent information that is not present in the table_data.
            - If the table_data does not fully answer the question, clearly state the limitation.

            Output:
            - A final response in plain natural language, as if you are explaining the result to a human user.
        """


        query_prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", user_que)]
        )
        messages = query_prompt_template.format_messages(
            user_que = user_que,
            dialect = self.db.dialect,
            tables = tables,
            schema = get_table_info_pg_str(self.db),
            table_data = table_data
        )

        raw_response = self.llm.invoke(messages)
        # Extract text (handles both string and message object cases)
        response_text = getattr(raw_response, "content", None) or str(raw_response)

        print(response_text)

        print()
        print()
        print()
        print("########################################################################################################")
        print("########################################################################################################")
        print("########################################################################################################")
        print()
        print()
        print()
        state["tables"].append(table_data)
        return state

    def write_query(self, state: State):
        system_message = """
            You are an expert SQL query generator with memory of previous interactions. Your task is to generate a syntactically correct {dialect} SQL query from the user's natural language question.

            {memory_context}

            CRITICAL RULES FOR MEMORY AND CONTEXT:
            1. ALWAYS pay close attention to the conversation context above.
            2. If the question contains pronouns (her, his, their, it, she, he, they), use the context to identify what they refer to.
            3. If the question refers to a person or entity mentioned in previous interactions, use that information.
            4. The resolved question should guide your SQL generation: {user_resolved_question}

            CRITICAL SQL RULES:
            1. Use ONLY the exact table names and column names provided in the schema below.
            2. Column names are case-sensitive — use exact capitalization as shown.
            3. Never assume or invent column names — only use those explicitly listed.
            4. Do NOT use SELECT * — always specify only the relevant columns.
            5. Unless the question explicitly requests more, limit the result to 10 rows.
            6. For date/time filtering or extraction, use correct functions per dialect:
            - SQLite: strftime('%Y', column), strftime('%m', column)
            - MySQL: YEAR(column), MONTH(column), DAY(column)
            - PostgreSQL: EXTRACT(YEAR FROM column), EXTRACT(MONTH FROM column)
            7. For text matching, use LIKE with `%` wildcards (e.g., WHERE name LIKE '%john%').
            8. When searching by name, always use the column named 'name' (not 'username', etc.).
            9. Do not use aliases, subqueries, or joins unless necessary to answer the question.
            10. Only include valid SQL syntax for the specified dialect.
            11. Use lowercase for text values in WHERE clauses since all text data is stored in lowercase.
            12. When searching by id, check the column name for 'user_id', 'student_id', etc., and use it exactly as shown in the schema.

            IMPORTANT:
            - ALWAYS consider the conversation context when interpreting the question.
            - If a pronoun or reference is unclear, look at the previous interactions to resolve it.
            - The resolved question "{user_resolved_question}" should be your primary guide.

            DATABASE SCHEMA:
            {schema}
            Here is the table data {table_data}

            Convert the following user question into a valid SQL query, considering the conversation context and resolved question.
            ORIGINAL QUESTION: {user_que}
            RESOLVED QUESTION: {user_resolved_question}

            """
        user_que = state.get("user_query", "")
        user_resolved_que = state.get("resolved_user_query", "")
        dialect = self.db.dialect
        schema = get_table_info_pg_str(self.db)
        if not user_que or not user_resolved_que:
            print("No question from user found")
            return
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
        
        raw_response = self.llm.invoke(messages)
        # Handle different response types
        if hasattr(raw_response, 'content'):
            if isinstance(raw_response.content, str):
                response_text = raw_response.content.strip()
            elif isinstance(raw_response.content, list):
                response_text = str(raw_response.content[0]).strip() if raw_response.content else ""
            else:
                response_text = str(raw_response.content).strip()
        else:
            response_text = str(raw_response).strip()

        match = re.search(r"```sql\s+(.*?)```", response_text, re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
        else:
            sql_query = response_text.strip()

        # Remove all newlines
        sql_query = sql_query.replace("\n", " ")  # replaces newlines with space
        # Or, if you want to remove them completely: sql_query = sql_query.replace("\n", "")

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
        error_msg = state.get("error", "")
        failed_query = state.get("sql_query", "")
        user_query = state.get("user_query", "")
        schema = get_table_info_pg_str(self.db)      # should be string of tables + columns
        table_data = state.get("tables", "")  # optional: few sample rows for each table

        if not error_msg:
            return state

        prompt = f"""
        You are an expert SQL assistant. 
        A SQL query failed when trying to answer the user's question.

        ---
        ❌ Error:
        {error_msg}

        🧑 User Question:
        {user_query}

        📝 Failed SQL Query:
        {failed_query}

        📑 Database Schema:
        {schema}

        📊 Sample Table Data (for context):
        {table_data}
        ---

        ✅ Task:
        - Analyze the error and the schema carefully.
        - Generate a corrected SQL query that will work with this schema and data.
        - Ensure the corrected query is syntactically valid and relevant to the user question.
        - Return **only** the SQL inside a fenced code block like:

        ```sql
        SELECT ...
        ```
        """

        raw_response = self.llm.invoke(prompt)
        response_text = getattr(raw_response, "content", str(raw_response))

        match = re.search(r"```sql\s+(.*?)```", response_text, re.DOTALL)
        fixed_query = match.group(1).strip() if match else response_text.strip()

        state["sql_query"] = fixed_query
        state["error"] = None
        print("[FixQuery] New SQL:", fixed_query)
        return state

    def generate_answer(self, state: State):
        system_message = """
            You are an Expert SQL Assistant. Your role is to provide clear, concise, and accurate answers to user queries based on SQL query results.
            You will be given results from sql query execution, make it in human understandable text
            Just give the result no further deviation from result
            Here is the result of the query execution {query_result}
            Here is the prev context too  {context}
            """
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
        raw_response = self.llm.invoke(messages)
        response_text = getattr(raw_response, "content", None) or str(raw_response)
        if not response_text: 
            print("Response could not be generated or parsed")
            return state
        state["response"] = response_text
        print("the generated response is ", state["response"])
        # Add it to the memory now
        memory = {state.get("user_query", ""): state.get("response", "")}
        save_user_memory(state.get("username", ""), memory, "email")
        return state

    def build_graph(self):
        builder = StateGraph(State)

        builder.add_node("add_memory_context", self.add_memory_context)
        builder.add_node("resolve_user_query", self.resolve_user_query)
        builder.add_node("identify_tables", self.identify_tables)
        builder.add_node("write_query", self.write_query)
        builder.add_node("execute_query", self.execute_query)
        builder.add_node("fix_query", self.fix_query)
        builder.add_node("generate_answer", self.generate_answer)

        # Main flow
        builder.add_edge("add_memory_context", "resolve_user_query")
        builder.add_edge("resolve_user_query", "identify_tables")
        builder.add_edge("identify_tables", "write_query")
        builder.add_edge("write_query", "execute_query")

        # Retry branch: if execute_query fails
        builder.add_conditional_edges(
            "execute_query",
            lambda state: "fix_query" if state.get("error") else "generate_answer",
            {"fix_query": "fix_query", "generate_answer": "generate_answer"}
        )
        builder.add_edge("fix_query", "execute_query")

        builder.set_entry_point("add_memory_context")
        return builder.compile()

    # def build_graph(self):
    #     # Build the graph here
    #     graph_builder = StateGraph(State)
    #     graph_builder.add_node("add_memory_context", self.add_memory_context)
    #     graph_builder.add_node("resolve_user_query", self.resolve_user_query)
    #     graph_builder.add_node("identify_tables", self.identify_tables)
    #     graph_builder.add_node("write_query", self.write_query)
    #     graph_builder.add_node("execute_query", self.execute_query)
    #     graph_builder.add_node("generate_answer", self.generate_answer)

    #     graph_builder.add_edge("add_memory_context", "resolve_user_query")
    #     graph_builder.add_edge("resolve_user_query", "identify_tables")
    #     graph_builder.add_edge("identify_tables", "write_query")
    #     graph_builder.add_edge("write_query", "execute_query")
    #     graph_builder.add_edge("execute_query", "generate_answer")
    #     graph_builder.set_entry_point("add_memory_context")

    #     return graph_builder.compile()
    


# initial_state = State(username="anurag", user_query="I have a doubt", resolved_user_query=None, sql_query=None, sql_query_response=None, response=None, error=None, context=None)

data_db_uri = config.DATA_DB_URI
memory_db_uri = config.MEMORY_DB_URI

agent = SQLAgent(data_db_uri, memory_db_uri)
initial_state = State(username="818", user_query="Show me the name of all items whose all document has been been completed that is they are not pending.", resolved_user_query=None, sql_query=None, sql_query_response=None, response=None, error=None, context=None, tables=None)
final_state = agent.graph.invoke(initial_state)
# print("The final state is ", final_state)
    print()
print()
print()
print("the extracted tables are ", final_state.get("tables", []))
print()
print()
print("The sql response is ", final_state.get("sql_query_response", ""))
print()
print()
print()
print("The response is ", final_state.get("response", ""))
print()
print()
print()