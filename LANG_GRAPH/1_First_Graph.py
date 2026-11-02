from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.3
)


# -------------------------------- Step 1 :- Define the schema --------------------------------
class graph_schema(TypedDict):
    name: str
    message: str


# -------------------------------- Step 2 :- Create the node functions --------------------------------


def welcome(state: graph_schema) -> dict:
    curr_name = state["name"]
    curr_message = state["message"]

    response = llm_mistral.invoke(f"my name is {curr_name}. {curr_message}").content

    return {
        "message": (
            f"Your message was {curr_message}.\nHere's my response:\n{response}"
        )
    }


# -------------------------------- Step 3 :- Create the state graph --------------------------------
from langgraph.graph import END, START, StateGraph

graph = StateGraph(graph_schema)
graph.add_node("welcome", welcome)
graph.add_edge(START, "welcome")
graph.add_edge("welcome", END)

# -------------------------------- Step 4 :- Compile the graph --------------------------------
first_graph = graph.compile()

# print(first_graph.get_graph().draw_mermaid())

# -------------------------------- Step 5 :- Calling the graph --------------------------------
print(
    first_graph.invoke({"name": "Mohit Lakra", "message": "How are you? "})["message"]
)
