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
            model="meta-llama/llama-4-scout",   # ✅ fixed model call
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
    def _call_llm_basic(self, system_message: str, user_que: str):
        """Helper to call OpenRouter API and return response_text"""
        completion = self.llm.chat.completions.create(
            model="meta-llama/llama-4-scout",
            # model="deepseek/deepseek-chat-v3.1",   # ✅ fixed model call
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

            You are a Query Resolver Assistant.
            Your job is to process user queries that are meant to fetch information from a structured database.

            You must interpret, clarify, and reconstruct informal or incomplete user questions into clear, contextually accurate, schema-aligned natural language queries — not SQL queries.

            🧩 INPUTS

            Database Dialect: {dialect}

            Database Schema: {schema}

            User Query: {user_que}

            Previous Context: {context}

            Table relationships: {get_table_relationships}

            🎯 OBJECTIVE

            Convert {user_que} into a cleaned, grammatically correct, contextually accurate, schema-aligned question written in natural language, which:

            Clearly describes the data request or intent

            Can later be translated into SQL

            Does not contain SQL syntax

            Is precise, unambiguous, and schema-valid

            🧭 RULES & INSTRUCTIONS
            1. Understand the User’s Intent

            The input may be incomplete, grammatically incorrect, or informal.

            Infer meaning logically but do not invent or assume data that isn’t implied.

            The output must preserve the user’s original intent as closely as possible.

            2. Follow Schema Strictly

            Align all references to the provided database schema.

            If the user says “project name,” rewrite it as “project title” (since the column is title in projects).

            Use the correct table and column names based on context.

            Avoid introducing non-existent attributes.

            3. Use Context for Reference Resolution

            If the query includes pronouns like he, she, it, that, this, they, or those, resolve them using {context}.

            Example:

            Context: “Show details of invoice 105.”

            User: “Show its items.”

            Resolved: “Show all invoice_items where invoice_id = 105.”

            4. Entity Normalization

            Normalize identifiers and codes based on conventions:

            “lot 1” → “lot-001”

            “invoice 2” → “invoice-002”

            Follow consistent normalization for clarity.

            5. Correct and Clarify Language

            Fix spelling mistakes and grammar.

            Expand shorthand or informal expressions:

            “emp salry” → “employee salary”

            “proj val” → “project total value”

            6. Maintain Logical Coherence

            Ensure the rewritten question logically matches the context and schema.

            If multiple possible interpretations exist, choose the most reasonable one aligned with the schema.

            7. Do NOT Output SQL

            The rewritten question must remain in natural language, not SQL.

            Example: “List all invoices created after January 2024,” not SELECT * FROM invoices....

            8. Preserve All Key Entities

            Do not drop or alter key fields, conditions, or identifiers (e.g., project IDs, lot numbers, supplier names, etc.).

            If filters or conditions are present, preserve them faithfully.

            9. Ambiguity Handling

            If a query is unclear, resolve it to the most plausible, schema-consistent interpretation.

            If still ambiguous, choose the most general interpretation without losing user intent.

            10. Formatting and Output

            Always output the result strictly in JSON format.

            Do not include explanations, reasoning, or extra commentary.

            Output JSON Format:

            {{
            "user_que": "<original user query>",
            "resolved_ques": "<cleaned, corrected, contextually accurate question>"
            }}

            💡 EXAMPLES
            
            Example 1

            User Query:
            give me detail of emp salry dept id 5
            Output:
            {{
            "user_que": "give me detail of emp salry dept id 5",
            "resolved_ques": "Get details of employee salaries where department_id = 5"
            }}

            Example 2
            
            Context:
            Get details of project titled Metro Rail Expansion
            User Query:
            show its lots
            Output:
            {{
            "user_que": "show its lots",
            "resolved_ques": "Show all lots associated with the project titled Metro Rail Expansion"
            }}

            Example 3

            User Query:
            what items are there in lot 1
            Output:
            {{
            "user_que": "what items are there in lot 1",
            "resolved_ques": "List all items associated with lot-001"
            }}

            Example 4
            User Query:
            show all rejected invoices for project 3
            Output:
            {{
            "user_que": "show all rejected invoices for project 3",
            "resolved_ques": "Show all invoices where project_id = 3 and status is 'rejected'"
            }}

            Example 5

            Context:
            Show details of supplier ABC Engineering
            User Query:
            show his invoices
            Output:
            {{
            "user_que": "show his invoices",
            "resolved_ques": "Show all invoices where supplier name is ABC Engineering"
            }}

            FINAL CHECKLIST
            - Use schema column names accurately
            - Use context to resolve references
            - Normalize entity naming (lot-001, invoice-002, etc.)
            - Return only JSON
            - Do not output SQL
            - Do not add commentary
            - Preserve meaning, accuracy, and structure



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
                context=state.get("context", ""),
                get_table_relationships = config.get_table_relationships(config.DATA_DB_URI)

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

            Your job is to identify all database tables that are relevant to answering a given user query based on the database schema, the resolved natural language question, and known relationships between tables.

            INPUTS YOU WILL RECEIVE

            You will be given the following:

            Database dialect: {dialect}
            Database schema: {schema} (includes all table names and columns)
            User query: {user_que} (the raw informal question)
            Resolved question: {resolved_ques} (clean, corrected natural language version of the query)
            Table relationships: {get_table_relationships} (foreign key mappings or logical relationships between tables)
            
            YOUR TASK
            Your goal is to determine which tables from the schema are directly or indirectly related to answering the user’s question. You must follow these steps:
            Identify explicit references
            Detect tables explicitly mentioned in the user query or resolved question (e.g., “invoice”, “project”, “lot”, “supplier”, etc.).
            Include implicitly related tables
            Even if a table is not mentioned, include it if it is necessary to resolve the question.
            Example: If user asks about “supplier invoices”, both suppliers and invoices are required, and since invoices have relationships with invoice_items, you should include that too.
            Trace relationships
            If a table connects to another through a foreign key or through projectId, lotId, invoiceId, departmentId, etc., include all connected tables that might hold relevant information.
            Example: Asking about a “lot” might also include related tables like lot_items, lot_revisions, invoices, and projects.
            Be inclusive, not exclusive
            When in doubt, include the table. Missing a relevant table is worse than including an extra one.
            Schema and relationship awareness
            Consider all major linking keys (projectId, lotId, invoiceId, supplierId, departmentId, clientId, etc.) to determine relatedness.
            You should automatically know which tables are connected through these keys (based on the schema and provided relationships).

            RELATIONSHIP LOGIC EXAMPLES
            You must always consider the following known relationship paths derived from the schema:

            projects ↔ lots ↔ lot_items ↔ master_items

            projects ↔ invoices ↔ invoice_items

            projects ↔ document_requirements ↔ document_alternatives ↔ documents

            projects ↔ departments ↔ project_departments

            projects ↔ clients

            projects ↔ suppliers (via items or invoices)

            projects ↔ inspection_agencies

            lots ↔ documents, email_logs, audit_logs, notifications

            invoices ↔ invoice_items, documents, suppliers, audit_logs

            documents ↔ document_requirements, document_alternatives, audit_logs, email_logs

            departments ↔ users ↔ notifications

            users ↔ audit_logs, notifications, email_logs

            suppliers ↔ item_suppliers ↔ master_items ↔ lot_items ↔ lots

            categories ↔ projects

            stakeholders ↔ lots

            master_items ↔ lot_items, item_suppliers, lot_revisions

            lot_revisions ↔ lots, projects, master_items

            Whenever any of these tables appear or are implied, include all linked tables in the relationship chain.

            CRITICAL RULES

            Output format must be STRICT JSON.
            NO explanations, notes, or natural text outside the JSON.
            Do NOT alter table names. Use them exactly as in the schema (case-sensitive).
            Do NOT output SQL queries.
            If no relevant tables can be found (which is rare), return an empty list.
            Always return JSON in the exact format below.

            OUTPUT FORMAT (MANDATORY)
            {{
            "user_que": "<original user query>",
            "tables": ["<table1>", "<table2>", "<table3>", ...]
            }}

            💡 EXAMPLES
            Example 1:

            User Input:
            give me detail of emp salry dept id 5
            Output:
            {{
            "user_que": "give me detail of emp salry dept id 5",
            "tables": ["employee", "salary", "department"]
            }}

            Example 2:

            User Input:
            show invoices of lot 1
            Output:
            {{
            "user_que": "show invoices of lot 1",
            "tables": ["lots", "invoices", "invoice_items", "projects", "suppliers", "documents"]
            }}

            Example 3:

            User Input:
            what documents were uploaded by user john for project A
            Output:
            {{
            "user_que": "what documents were uploaded by user john for project A",
            "tables": ["documents", "users", "projects", "departments", "document_requirements", "document_alternatives", "audit_logs"]
            }}

            Example 4:

            User Input:
            get total value and progress of project 12
            Output:
            {{
            "user_que": "get total value and progress of project 12",
            "tables": ["projects", "lots", "lot_items", "invoices", "invoice_items", "master_items", "clients", "categories", "suppliers"]
            }}

            Example 5:

            User Input:
            show me all revision details for lot 2
            Output:
            {{
            "user_que": "show me all revision details for lot 2",
            "tables": ["lots", "lot_revisions", "projects", "master_items", "departments", "audit_logs"]
            }}

            Example 6:

            User Input:
            which suppliers are linked to project alpha
            Output:
            {{
            "user_que": "which suppliers are linked to project alpha",
            "tables": ["projects", "suppliers", "item_suppliers", "master_items", "lots", "lot_items", "invoices"]
            }}

            Example 7:

            User Input:
            show me all notifications for user rohit related to project 5
            Output:
            {{
            "user_que": "show me all notifications for user rohit related to project 5",
            "tables": ["notifications", "users", "projects", "lots", "departments", "email_logs"]
            }}

            FINAL NOTE
            You are a STRICT, schema-aware table detection system.
            Always err on the side of inclusion — when in doubt, add the table.
            Never output anything except the required JSON format.
        """

        # ✅ Changed: Use OpenRouter call
        response_text = self._call_llm_basic(
            system_message.format(
                dialect=self.db.dialect,
                schema=get_table_info_pg_str(self.db),
                user_que=user_que,
                resolved_ques=state.get("resolved_user_query", ""),
                get_table_relationships=config.get_table_relationships(config.DATA_DB_URI)
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
        # print("the table data is ", table_data)
        print("########################################################################################################")
        print("########################################################################################################")
        print("########################################################################################################")
        print()

        # ✅ Changed: Generate NL answer with OpenRouter
        system_prompt = """
            You are an Expert SQL Assistant whose ONLY job is to convert structured database information into clear, natural, human-readable answers.

            INPUTS YOU WILL RECEIVE

            You will be given the following inputs:

            User query: {user_que} — the natural language question originally asked by the user.

            Database dialect: {dialect} — the SQL dialect being used (e.g., PostgreSQL, MySQL, SQLite).

            Relevant tables: {tables} — the list of tables identified as relevant to answering the query.

            Table schema: {schema} — includes table names, columns, and data types for the relevant tables.

            Table data: {table_data} — the actual rows retrieved from the database for these tables.

            Table relationships: {get_table_relationships} — known links between tables (foreign keys, shared fields, etc.).

            YOUR TASK

            Your objective is to read the given table_data carefully and generate a concise, human-like explanation that answers the user’s question as accurately as possible.

            HOW TO THINK AND RESPOND

            Understand the user intent

            Read {user_que} carefully and determine what the user wants to know.

            Focus on what information they are asking for, not on how to retrieve it.

            Use only the provided table_data

            Your answer should come strictly from {table_data}.

            Do not infer or assume data that is missing.

            If the table_data doesn’t contain enough information, clearly mention that the available data is incomplete.

            Generate a natural, conversational response

            Your answer must sound like a short, clear human explanation, not a machine output.

            Avoid robotic phrasing or repetition.

            Use full, meaningful information

            If the user asks about an entity (e.g., a project, invoice, supplier, document), show the relevant descriptive fields such as name, title, description, or invoiceNumber — not just id.

            If multiple rows are present, summarize them naturally (e.g., counts, lists, grouped summaries).

            Handle relationships between tables

            Use {get_table_relationships} to connect related data logically.

            Example: If a project connects to lots and invoices, combine relevant data into one smooth summary.

            Stay concise and precise

            Keep the tone natural, like a human summarizing a result.

            Avoid unnecessary technical or explanatory text (no "The query returns..." or "Based on SQL...").

            But do not trim meaningful data — include all actual result details in a short, readable format.

            Be clear about limitations

            If the data is partial or incomplete, explicitly say so in one short line.

            Example: “Only partial data is available for this project.”

            Never produce SQL

            You must not output SQL code, table structures, or any technical commentary.

            You are a human-style explainer, not a SQL generator.

            ⚙️ RESPONSE FORMAT

            Your output must be in plain natural language only, with no JSON, code blocks, or metadata.
            It should read like a sentence or short paragraph — the final conversational answer to the user.

            💡 EXAMPLES
            Example 1:

            User Query:
            show invoices of lot 1

            Table Data (simplified):

            invoiceNumber	totalAmount	status	lotId	supplierId	invoiceDate
            INV-001	15000	Approved	lot-001	sup-101	2024-05-10
            INV-002	22000	Pending	lot-001	sup-102	2024-06-02

            Output:

            There are two invoices for Lot 1 — Invoice INV-001 approved for ₹15,000 on May 10, 2024, and Invoice INV-002 pending for ₹22,000 on June 2, 2024.

            Example 2:

            User Query:
            who uploaded the documents for project A

            Table Data (simplified):

            documentName	uploadedBy	projectTitle
            Invoice.pdf	John	Project A
            PO.pdf	Sarah	Project A

            Output:

            For Project A, John uploaded Invoice.pdf and Sarah uploaded PO.pdf.

            Example 3:

            User Query:
            get total value and progress of project 12

            Table Data (simplified):

            projectTitle	totalValue	progress
            Metro Project 12	8,500,000	72

            Output:

            Project “Metro Project 12” has a total value of ₹8.5 million and is 72% complete.

            Example 4:

            User Query:
            show me all revision details for lot 2

            Table Data (simplified):

            itemDescription	originalQuantity	revisedQuantity	reason	status
            Steel Rods	100	120	Additional supply	Approved
            Screws	500	450	Overcount correction	Approved

            Output:

            Lot 2 had two revisions — Steel Rods increased from 100 to 120 due to additional supply, and Screws reduced from 500 to 450 to correct overcount. Both revisions were approved.

            Example 5:

            User Query:
            what notifications were sent to user rohit

            Table Data (simplified):

            title	message	isRead
            Invoice Approval	Your invoice INV-001 was approved	true
            New Document	A new PO.pdf was uploaded	false

            Output:

            Rohit received two notifications — one confirming invoice INV-001 approval (read) and another about a new document upload (unread).

            Example 6:

            User Query:
            show supplier details for project alpha

            Table Data (simplified):

            supplierName	email	status
            Apex Metals	sales@apex.com
                Active
            Global Steel	contact@globalsteel.com
                Active

            Output:

            Project Alpha involves two suppliers — Apex Metals (sales@apex.com
            ) and Global Steel (contact@globalsteel.com
            ), both currently active.

            CRITICAL RULES

            Do NOT output SQL, JSON, or code.

            Do NOT explain how the answer was derived.

            Do NOT assume or invent information.

            Always answer in fluent English, short but complete.

            Always use context from table_data and relationships.

            If data is incomplete, say so briefly.

            Never output anything except the human-readable final answer.

            SUMMARY OF BEHAVIOR

            You are a professional, human-like summarizer of structured data results.
            Your goal is to convert raw SQL result rows into smooth, natural English answers,
            while maintaining full accuracy and respecting the provided data context.
        """
        response_text = self._call_llm_basic(
            system_prompt.format(
                user_que=user_que,
                dialect=self.db.dialect,
                tables=tables,
                schema=get_table_info_pg_str(self.db),
                table_data=table_data,
                get_table_relationships = config.get_table_relationships(config.DATA_DB_URI)
            ),
            user_que
        )

        print(response_text)
        state["response"] = response_text
        memory = {state.get("user_query", ""): state.get("response", "")}
        save_user_memory(state.get("username", ""), memory, "email")
        print("Saved the user memory in the db")
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
        # builder.add_node("write_query", self.write_query)
        # builder.add_node("execute_query", self.execute_query)
        # builder.add_node("fix_query", self.fix_query)
        # builder.add_node("generate_answer", self.generate_answer)

        builder.add_edge("add_memory_context", "resolve_user_query")
        builder.add_edge("resolve_user_query", "identify_tables")
        # builder.add_edge("identify_tables", "write_query")
        # builder.add_edge("write_query", "execute_query")

        # builder.add_conditional_edges(
        #     "execute_query",
        #     lambda state: "fix_query" if state.get("error") else "generate_answer",
        #     {"fix_query": "fix_query", "generate_answer": "generate_answer"}
        # )
        # builder.add_edge("fix_query", "execute_query")

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
