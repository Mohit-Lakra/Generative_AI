import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

if os.environ["MISTRAL_API_KEY"]:
    print("Mistral API Key is set.")
else:
    raise ValueError("Mistral API Key is not set.")

from langchain_core.messages import HumanMessage

llm = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.3
)

from operator import add
from typing import Annotated, TypedDict


class graph_schema(TypedDict):
    messages_manual: list
    # Using Reducer in Langgraph
    messages_auto: Annotated[list, add]


from langchain_core.messages import AIMessage


def create_post(state: graph_schema) -> graph_schema:

    messages_manual = state["messages_manual"]
    response_manual = llm.invoke(messages_manual).content
    response__manual_ai = AIMessage(content=response_manual)
    state["messages_manual"] = messages_manual + [response__manual_ai]

    messages_auto = state["messages_auto"]
    response_auto = llm.invoke(messages_auto).content
    response_auto_ai = AIMessage(content=response_auto)
    state["messages_auto"] = [response_auto_ai]

    return state


def curate_post(state: graph_schema) -> graph_schema:

    messages_manual = state["messages_manual"]

    messages_manual.append(
        HumanMessage(content="Please curate and improve the previous post.")
    )

    response_manual = llm.invoke(messages_manual).content
    response__manual_ai = AIMessage(content=response_manual)

    state["messages_manual"] = messages_manual + [response__manual_ai]

    messages_auto = state["messages_auto"]

    messages_auto.append(
        HumanMessage(content="Please curate and improve the previous post.")
    )

    response_auto = llm.invoke(messages_auto).content
    response_auto_ai = AIMessage(content=response_auto)

    state["messages_auto"] = messages_auto + [response_auto_ai]

    return state


from langgraph.graph import END, START, StateGraph

graph = StateGraph(graph_schema)

graph.add_node("create_post", create_post)
graph.add_node("curate_post", curate_post)

graph.add_edge(START, "create_post")
graph.add_edge("create_post", "curate_post")
graph.add_edge("curate_post", END)

messages_graph = graph.compile()

messages_graph.invoke(
    {
        "messages_manual": [
            HumanMessage(content="The importance of data privacy in the digital age")
        ],
        "messages_auto": [
            HumanMessage(content="The importance of data privacy in the digital age")
        ],
    }
)
