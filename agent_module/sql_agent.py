import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import re
import sqlite3
from typing import Dict, Any, Optional
from openai import OpenAI   # ✅ Changed import: using OpenRouter/OpenAI SDK
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, NotRequired
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

        # ✅ Changed: initialize OpenRouter client instead of LangChain ChatOpenAI
        self.llm = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER
        )

        self.db = SQLDatabase.from_uri(data_db_uri)
        # print(self.db.dialect)
        # User memory db object
        self.user_memory_db_object = SQLDatabase.from_uri(user_memory_db_uri)
        # Build the graph
        self.graph = self.build_graph()

    def _call_llm(self, system_message: str, user_que: str):
        """Helper to call OpenRouter API and return response_text"""
        completion = self.llm.chat.completions.create(
            # model="meta-llama/llama-4-scout",   # ✅ fixed model call
            model="openai/gpt-4.1",
            # model="qwen/qwen3-coder-plus",
            # model="x-ai/grok-code-fast-1",
            # model="anthropic/claude-sonnet-4.5",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_que},
            ],
        )
        raw_text = completion.choices[0].message.content.strip()
        raw_response = raw_text
        response_text = getattr(raw_response, "content", None) or str(raw_response)
        return response_text
    def _call_llm_basic(self, system_message: str, user_que: str):
        """Helper to call OpenRouter API and return response_text"""
        completion = self.llm.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",   # ✅ fixed model call
            # model="openai/gpt-4.1",
            # model="qwen/qwen3-coder-plus",
            # model="x-ai/grok-code-fast-1",
            # model="anthropic/claude-sonnet-4.5",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_que},
            ],
        )
        raw_text = completion.choices[0].message.content.strip()
        raw_response = raw_text
        response_text = getattr(raw_response, "content", None) or str(raw_response)
        return response_text

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
            If the user query have references like he/his/her/she/it/that etc like these, take reference from the the context provided to you to resolve and re-write the question
            Keep in mind if user is asking about say lot 1 then consider it as lot-001 etc
            Output Format
            Always return the result in strict JSON format with the following structure:
            {{
            "user_que": "<original user query>",
            "resolved_ques": "<cleaned, corrected query>"
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

        # ✅ Changed: Use OpenRouter API call instead of self.llm.invoke
        response_text = self._call_llm_basic(
            system_message.format(
                dialect=self.db.dialect,
                schema=get_table_info_pg_str(self.db),
                user_que=user_que,
                context=state.get("context", "")
            ),
            user_que
        )

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
        system_message = """
            You are a STRICT Table Identifier Assistant. 
            You will be given:
            - The database dialect: {dialect}
            - The complete database schema: {schema}
            - A natural language user query: {user_que}
            - Resolved question: {resolved_ques}

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

        # ✅ Changed: Use OpenRouter call
        response_text = self._call_llm_basic(
            system_message.format(
                dialect=self.db.dialect,
                schema=get_table_info_pg_str(self.db),
                user_que=user_que,
                resolved_ques=state.get("resolved_user_query", "")
            ),
            user_que
        )

        response_json = {}
        tables = []
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
            tables = response_json.get("tables", [])
        if not tables:
            print("No relevant tables found")
            state["tables"] = []
        else:
            state["tables"] = tables
        tables = list(set(tables))

        # fetch table data (same logic as before)
        table_data = []
        for table in tables:
            try:
                query = f"SELECT * FROM {table};"
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
        print("the table data is ", table_data)
        print("########################################################################################################")
        print("########################################################################################################")
        print("########################################################################################################")
        print()

        # ✅ Changed: Generate NL answer with OpenRouter
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
        response_text = self._call_llm_basic(
            system_prompt.format(
                user_que=user_que,
                dialect=self.db.dialect,
                tables=tables,
                schema=get_table_info_pg_str(self.db),
                table_data=table_data
            ),
            user_que
        )

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

    # --- similarly, replace in write_query, fix_query, generate_answer ---
    # (due to space I’ll keep them same structure: just replace self.llm.invoke with self._call_llm)
    # rest of your code remains unchanged ...

    # The rest of the methods follow the same pattern:
    # - build system_message
    # - call self._call_llm(system_message, user_que)
    # - process response_text
    

    def write_query(self, state: State):
        system_message = """
            You are an expert PostgreSQL SQL query generator.

            Your job is to transform a user’s natural language request into a syntactically correct PostgreSQL query.  
            You must always respect the schema, relationships, and context provided.  

            ━━━━━━━━━━━━━━
            🔹 Context & Resolution Rules

            1. Always use {memory_context} to resolve pronouns, references, and ambiguities.  
            - If the user says “their invoices” → resolve to the appropriate user/project/entity from memory.  
            - If the user refers to “that project” → resolve it from the last conversation turn.

            2. Always validate the meaning of {user_query} against {resolved_query}.  
            - {resolved_query} is authoritative.  
            - {user_query} is secondary input.

            ━━━━━━━━━━━━━━
            🔹 PostgreSQL SQL Generation Rules

            **Schema Compliance**  
            - Use only exact table names and case-sensitive column names from {schema}.  
            - Never invent or assume columns.  
            - If column is camelCase, always wrap in quotes (e.g., "selectedSupplierId").  

            **Column Selection**  
            - Never use `SELECT *`.  
            - Only select necessary columns relevant to the query.  
            - Use meaningful aliases (e.g., `AS supplier_name`, `AS project_title`).  

            **Date & Time**  
            - Extract year → `EXTRACT(YEAR FROM column)`  
            - Extract month → `EXTRACT(MONTH FROM column)`  
            - Extract day → `EXTRACT(DAY FROM column)`  

            **Text Matching**  
            - Always use case-insensitive matching with `ILIKE`.  
            - Example: `WHERE u.name ILIKE '%john%'`  

            **ID Filtering**  
            - Always filter by correct primary/foreign key columns (`userId`, `projectId`, etc.)  
            - Do not assume column names—use only schema definitions.  

            **Joins & Relationships**  
            - Always derive joins from {relationships}.  
            - Use INNER JOIN by default.  
            - Use LEFT JOIN only if relationship is optional (e.g., audit logs).  
            - Join foreign key → primary key directly.  

            **Query Structure**  
            - Keep queries minimal and clear.  
            - No unnecessary aliases, subqueries, or joins.  
            - Always order results when user implies chronology (e.g., `ORDER BY createdAt DESC`).  
            - When aggregating, always use `GROUP BY` correctly.

            ━━━━━━━━━━━━━━
            🔹 Inputs You Will Receive

            - {schema} → PostgreSQL table definitions with column names & types.  
            - {relationships} → Explicit FK → PK mappings.  
            - {table_data} (optional) → Example rows (for context only; never hardcode values).  
            - {memory_context} → Conversation history & entity resolution memory.  
            - {user_query} → Raw natural language request.  
            - {resolved_query} → Clarified and authoritative interpretation of user request.  

            ━━━━━━━━━━━━━━
            🔹 Output Rules

            - Always return only the **SQL query**.  
            - No explanation, no commentary, no extra formatting.  
            - Ensure strict PostgreSQL compliance.  

            ━━━━━━━━━━━━━━
            🔹 Examples

            Example 1
            user_query:
            Show me all invoices with their supplier names, project titles, and the user who created them.

            SQL_query:

            SELECT i."id" AS invoice_id,
                i."invoiceNumber",
                i."totalAmount",
                s."name" AS supplier_name,
                p."title" AS project_title,
                u."name" AS created_by
            FROM invoices i
            JOIN suppliers s ON i."supplierId" = s."id"
            JOIN projects p ON i."projectId" = p."id"
            JOIN users u ON i."createdById" = u."id";

            Example 2
            user_query:
            List all documents uploaded for a project, along with the department name, uploader name, and related lot number.

            SQL_query:

            SELECT d."id" AS document_id,
                d."originalName",
                dep."name" AS department_name,
                u."name" AS uploaded_by,
                l."lotNumber"
            FROM documents d
            JOIN departments dep ON d."departmentId" = dep."id"
            JOIN users u ON d."uploadedById" = u."id"
            JOIN lots l ON d."lotId" = l."id"
            WHERE d."projectId" = 'some_project_id';

            Example 3
            user_query:
            Get all lot items with their master item description, selected supplier name, and related project title.

            SQL_query:

            SELECT li."id" AS lot_item_id,
                li."quantity",
                mi."description" AS master_item_description,
                s."name" AS selected_supplier,
                p."title" AS project_title
            FROM lot_items li
            JOIN master_items mi ON li."masterItemId" = mi."id"
            JOIN suppliers s ON li."selectedSupplierId" = s."id"
            JOIN lots l ON li."lotId" = l."id"
            JOIN projects p ON l."projectId" = p."id";

            Example 4
            user_query:
            Show me all audit log entries with their related project title, lot number, invoice number, and the user who performed the action.

            SQL_query:

            SELECT a."id" AS audit_id,
                a."action",
                a."entity",
                u."name" AS user_name,
                p."title" AS project_title,
                l."lotNumber",
                i."invoiceNumber",
                a."createdAt"
            FROM audit_logs a
            LEFT JOIN users u ON a."userId" = u."id"
            LEFT JOIN projects p ON a."projectId" = p."id"
            LEFT JOIN lots l ON a."lotId" = l."id"
            LEFT JOIN invoices i ON a."invoiceId" = i."id"
            ORDER BY a."createdAt" DESC;

            Example 5
            user_query:
            Get all invoice items with their invoice number, master item description, lot number, and supplier name.

            SQL_query:

            SELECT ii."id" AS invoice_item_id,
                ii."description",
                ii."quantity",
                ii."totalPrice",
                inv."invoiceNumber",
                mi."description" AS master_item,
                l."lotNumber",
                s."name" AS supplier_name
            FROM invoice_items ii
            JOIN invoices inv ON ii."invoiceId" = inv."id"
            JOIN master_items mi ON ii."masterItemId" = mi."id"
            JOIN lot_items li ON ii."lotItemId" = li."id"
            JOIN lots l ON li."lotId" = l."id"
            JOIN suppliers s ON inv."supplierId" = s."id";

            Example 6
            user_query:
            Find all projects with their category name, client name, inspection agency, and manager name.

            SQL_query:

            SELECT p."id" AS project_id,
                p."title",
                c."name" AS category_name,
                cl."name" AS client_name,
                ia."name" AS inspection_agency,
                u."name" AS manager_name,
                p."startDate",
                p."endDate"
            FROM projects p
            JOIN categories c ON p."categoryId" = c."id"
            JOIN clients cl ON p."clientId" = cl."id"
            JOIN inspection_agencies ia ON p."inspectionAgencyId" = ia."id"
            JOIN users u ON p."managerId" = u."id";


            These examples cover joins across multiple tables, foreign key relationships, filters, and ordering

            ━━━━━━━━━━━━━━
            🔹 Task

            1. Interpret {user_query} in light of {memory_context}.  
            2. Confirm intent with {resolved_query}.  
            3. Generate a correct PostgreSQL SQL query that strictly follows {schema}, {relationships}, and rules above.  
            4. Return only the SQL query.  
            """

        user_que = state.get("user_query", "")
        user_resolved_que = state.get("resolved_user_query", "")
        dialect = self.db.dialect
        schema = get_table_info_pg_str(self.db)
        relationships = config.get_table_relationships(config.DATA_DB_URI)
        if not user_que or not user_resolved_que:
            print("No question from user found")
            return
        
        response_text = self._call_llm(
            system_message.format(
                dialect=dialect,
                schema=schema,
                user_query=user_que,
                resolved_query=user_resolved_que,
                memory_context=state.get("context", "No prior context."),
                table_data=state.get("tables", ""),
                relationships=relationships
            ),
            user_que
        )
        print()
        print()
        print()
        print()
        print("The raw response from llm is ", response_text)
        # Handle different response types
        # if hasattr(raw_response, 'content'):
        #     if isinstance(raw_response.content, str):
        #         response_text = raw_response.content.strip()
        #     elif isinstance(raw_response.content, list):
        #         response_text = str(raw_response.content[0]).strip() if raw_response.content else ""
        #     else:
        #         response_text = str(raw_response.content).strip()
        # else:
        #     response_text = str(raw_response).strip()

        match = re.search(r"```sql\s+(.*?)```", response_text, re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
        else:
            sql_query = response_text.strip()

        # Remove all newlines
        sql_query = sql_query.replace("\n", " ")  # replaces newlines with space
        # Or, if you want to remove them completely: sql_query = sql_query.replace("\n", "")

        state["sql_query"] = sql_query

        print()
        print()
        print()
        print()
        print("The generated query is ", sql_query)
        print()
        print()
        print()
        print()

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


    def fix_query(self, state: dict):
        """
        Attempt to fix a failed SQL query by passing error details, user query, schema, and sample data
        to the LLM. Updates the state with a corrected SQL query if possible.
        """

        error_msg = state.get("error", "")
        failed_query = state.get("sql_query", "")
        user_query = state.get("user_query", "")
        schema = get_table_info_pg_str(self.db)  # should return a string of tables + columns
        relationships = config.get_table_relationships(config.DATA_DB_URI)  # optional: FK/PK mappings
        table_data = state.get("tables", "")  # optional: few sample rows for each table

        # If no error, return state unchanged
        if not error_msg:
            return state

        # Construct a very detailed prompt
        prompt = f"""
            You are an **expert PostgreSQL SQL assistant**. 
            A SQL query has failed when answering a user’s natural language question. 
            Your job is to carefully analyze the error, schema, and context, then generate a corrected SQL query.

            ---
            ❌ Error Message:
            {error_msg}

            🧑 User Question:
            {user_query}

            📝 Failed SQL Query:
            {failed_query}

            📑 Database Schema:
            {schema}

            🔗 Table Relationships (FK → PK mappings):
            {relationships}

            📊 Sample Table Data (for context only, do not invent values):
            {table_data}
            ---

            ✅ Task:
            Follow these rules STRICTLY when producing the corrected query:

            1. **Case Sensitivity**
            - Use table and column names exactly as they appear in the schema.
            - Do not lowercase or uppercase names unless the schema defines them that way.

            2. **Schema Compliance**
            - Only use table names and column names from `{schema}`.
            - Do not invent or assume new fields.

            3. **No Wildcards**
            - Never use `SELECT *`. Always explicitly select only the required columns.

            4. **Result Limiting**
            - Unless the user explicitly requests more, dont limit the results.

            5. **Joins**
            - Use `{relationships}` to determine valid joins.
            - Foreign keys must join to their corresponding primary keys.
            - Use `INNER JOIN` by default; `LEFT JOIN` only when the relationship is optional.

            6. **Text Matching**
            - Always use `ILIKE` for case-insensitive text search.
            - Wrap user-provided search terms in `%...%` wildcards unless exact match is needed.

            7. **Date/Time Handling**
            - Use PostgreSQL syntax only:
                - `EXTRACT(YEAR FROM column)`
                - `EXTRACT(MONTH FROM column)`
                - `EXTRACT(DAY FROM column)`
                - `DATE_TRUNC('month', column)` when grouping by month.

            8. **ID and Key Usage**
            - Always filter by the exact ID column name defined in schema (`userId`, `projectId`, etc.).
            - Never assume generic `id` unless explicitly defined.

            9. **Aggregations (if needed)**
            - Use correct SQL aggregation (`COUNT`, `SUM`, `AVG`, etc.).
            - Always `GROUP BY` non-aggregated columns.

            10. **Query Clarity**
                - Do not add unnecessary subqueries, aliases, or joins.
                - Keep the query as simple and direct as possible while answering the question.

            11. **Error Correction**
                - Fix invalid column names, incorrect joins, or syntax issues that caused the original error.
                - Ensure the new query is executable in PostgreSQL.

            12. **Final Output**
                - Return **only the corrected SQL query** inside a fenced code block like:

            ```sql
            SELECT ...
            ```
        """


        # Call LLM
        response_text = self._call_llm(prompt, user_query)

        # Extract the corrected SQL safely
        fixed_query = None
        match = re.search(r"```sql\s+(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            fixed_query = match.group(1).strip()
        else:
            # fallback: try to guess SQL if no code block was returned
            fixed_query = response_text.strip()

        # Update state
        state["sql_query"] = fixed_query
        state["error"] = None

        print("\n[FixQuery] Original Query:\n", failed_query)
        print("\n[FixQuery] Error:\n", error_msg)
        print("\n[FixQuery] Corrected Query:\n", fixed_query, "\n")

        return state


    def generate_answer(self, state: State):
        system_message = """
            You are an Expert SQL Assistant. Your role is to provide clear, concise, and accurate answers to user queries based on SQL query results.
            You will be given results from sql query execution, make it in human understandable text
            Just give the result no further deviation from result
            Here is the result of the query execution {query_result}
            Here is the prev context too  {context}
            """
        user_query = state.get("user_query", "")
        sql_query_response = state.get("sql_query_response", "")
        context = state.get("context", "")
        if not user_query or not sql_query_response:
            print("No user query or SQL response found for generating answer")
            return state

        response_text = self._call_llm_basic(
            system_message.format(
                query_result=sql_query_response,
                context=context
            ),
            user_query
        )
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

        builder.add_edge("add_memory_context", "resolve_user_query")
        builder.add_edge("resolve_user_query", "identify_tables")
        builder.add_edge("identify_tables", "write_query")
        builder.add_edge("write_query", "execute_query")

        builder.add_conditional_edges(
            "execute_query",
            lambda state: "fix_query" if state.get("error") else "generate_answer",
            {"fix_query": "fix_query", "generate_answer": "generate_answer"}
        )
        builder.add_edge("fix_query", "execute_query")

        builder.set_entry_point("add_memory_context")
        return builder.compile()


# initial_state = State(username="anurag", user_query="I have a doubt", resolved_user_query=None, sql_query=None, sql_query_response=None, response=None, error=None, context=None)

# data_db_uri = config.DATA_DB_URI
# memory_db_uri = config.MEMORY_DB_URI

# agent = SQLAgent(data_db_uri, memory_db_uri)
# initial_state = State(username="rrfgfglwds", user_query="Get the number of documents per client, showing only clients with more than 10 total documents.", resolved_user_query=None, sql_query=None, sql_query_response=None, response=None, error=None, context=None, tables=None)
# final_state = agent.graph.invoke(initial_state)

# print()
# print()
# print()
# print("The sql response is ", final_state.get("sql_query_response", ""))
# print()
# print("The response is ", final_state.get("response", ""))
# print()
