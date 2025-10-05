from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_toolkits import create_sql_agent
from langchain.agents import AgentExecutor
from langchain_groq import ChatGroq
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent_module.config import DATA_DB_URI

# Load environment variables
load_dotenv()


# Connect to Chinook DB
db = SQLDatabase.from_uri(DATA_DB_URI)
print(db.dialect)
print(db.get_usable_table_names())



# Initialize Groq LLM
llm = ChatGroq(
    api_key="---",
    model="llama-3.1-8b-instant"
)


# Memory to retain conversation context
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
tool_node = ToolNode(tools)
print('tools: ', [tool.name for tool in tools])

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
db_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")


print(list_tables_tool.invoke(""))

builder = StateGraph(MessagesState)
chat_history = memory.load_memory_variables({})["chat_history"]

def get_sql_prompt(schema: str):
    return ChatPromptTemplate.from_messages([
        SystemMessage(
f"""
You are a helpful AI assistant with access to a SQLite database. Here is the schema of the database:

{schema}
There will always be tables in the database that you can query. So, always find the most relevant table and then proceed. Never assume any table or column.
Your job is to:
1. Understand the user's question.
2. Decide whether an SQL query is needed.
3. If so, generate the correct SQL query using the schema above.
4. Use the appropriate tool to run the SQL query.
5. Respond to the user with:
    a. A natural language explanation.
    b. The SQL query you used.
    c. The query result.
6. Also take help from the previous conversation {chat_history} to answer the present question.
Guidelines:
- Never assume a column exists — only use columns from the schema.
- Be concise and accurate.
- Use JOINs appropriately if data spans multiple tables.
"""
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

schema_info = get_schema_tool.invoke("")


prompt = get_sql_prompt(schema_info)
llm_with_tools = prompt | llm.bind_tools(tools)

def sql_llm_node(state: MessagesState):

    runnable_input = {
        "messages": state["messages"],
    }

    response = llm_with_tools.invoke(runnable_input)

    new_state = {"messages": state["messages"] + [response]}

    last_human_message = state["messages"][-1]
    memory.save_context({"input": last_human_message.content}, {"output": response.content}) 

    return new_state

builder.add_node("llm", sql_llm_node)
builder.add_node("tool", tool_node)

# Define the entry point
builder.add_edge(START, "llm")

# Defining Conditional Edges
builder.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "__end__": END,
        "tools": "tool"
    }
)
builder.add_edge("tool", "llm")


# Compile the graph
graph = builder.compile()

def run_sql_agent(question: str):
    print(f"USER QUERY:\n{question}")

    input_message = HumanMessage(content=question)
    tool_result_printed = False 
    initial_state = {"messages": [input_message]}


    for event in graph.stream(initial_state, stream_mode="values"):
        messages = event.get('messages')
        if not messages:
            continue 

        last_msg = messages[-1]

        if isinstance(last_msg, AIMessage):
            print("\nResponse:")

            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                 print("\nGENERATED TOOL CALLS:")
                 for tool_call in last_msg.tool_calls:
                     print(f"Tool: {tool_call.get('name')}")
                     print(f"Args: {tool_call.get('args')}")
                     if tool_call.get('name') == 'sql_db_query' and 'query' in tool_call.get('args', {}):
                         sql_query = tool_call['args']['query']
                         print("\nGENERATED SQL QUERY:")
                         print(sql_query)
            else:
                 print(last_msg.content)

        elif isinstance(last_msg, ToolMessage) and not tool_result_printed:
            print("\nSQL RESULT:")
            print(last_msg.content)
            tool_result_printed = True  


# Graph structure
# from IPython.display import Image, display
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     pass


# Agent Call
run_sql_agent("""Show me the name of all items. """)

# ---END---