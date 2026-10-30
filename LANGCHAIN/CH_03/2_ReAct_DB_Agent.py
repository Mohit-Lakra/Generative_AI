from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

load_dotenv()

# 1. Configure Mistral
llm = init_chat_model(
    model="ministral-3b-latest",
    model_provider="mistralai",
    temperature=0,
    disable_streaming=True,
)

# 2. Locate database
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "SalesDB" / "sales.db"

if not DB_PATH.is_file():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

print("Tables:", db.get_usable_table_names())

# 3. Create SQL tools
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm,
)

tools = toolkit.get_tools()

# 4. Create modern agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a SQL database assistant.

Your job is to answer questions using the database.

Rules:
- Inspect available tables and relevant schemas.
- Generate correct SQLite queries.
- Execute queries using the SQL tools.
- Use actual query results for your answers.
- Never invent database values.
- Never modify the database.
- Only use SELECT queries.
- If a query fails, inspect the error and correct it.
""",
)

# 5. Execute agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "How much sale we made for Wireless Mouse?",
            }
        ]
    },
    config={"recursion_limit": 20},
)

# 6. Print final answer
print("\nFINAL ANSWER:")
print(response["messages"][-1].content)
