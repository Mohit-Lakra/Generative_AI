import wikipedia
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_agent

load_dotenv()

llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.3
)

# ================================== TOOLS ==================================

# -------------------------------- Tool 1 - News Search --------------------------------

search_tool = DuckDuckGoSearchRun()

# -------------------------------- Tool 2 - Wikipedia Search --------------------------------

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(wiki_client=wikipedia)
)


# -------------------------------- Tool 3 - Custom Enterprise Tool --------------------------------
@tool
def enterprise_tool(query: str) -> str:
    """
    This tool is used to send the email to employees
    """
    return "EMAIL SENT"


Toolkit = [search_tool, wikipedia_tool, enterprise_tool]


# -------------------------------- ReAct Agent --------------------------------

agent = create_agent(llm_mistral, tools=Toolkit)


# -------------------------------- ReAct Agent invoke with stream --------------------------------

query = "Give me top 5 latest news about stock market and make tool call 5 times"

events = agent.stream({"messages": [HumanMessage(content=query)]}, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()
