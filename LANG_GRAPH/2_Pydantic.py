from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.3
)

from pydantic import BaseModel, Field


class graph_schema(BaseModel):
    topic: str = Field(description="The topic of the graph")
    post: str = Field(description="The LinkedIn post content")
    curated_post: str = Field(description="The curated LinkedIn post content")


def create_post(schema: graph_schema) -> graph_schema:
    state = schema.model_dump()
    topic = state["topic"]
    post = llm_mistral.invoke(f"Write a LinkedIn post about {topic}").content
    state["post"] = post
    state = graph_schema.validate(state)
    return state


def curate_post(schema: graph_schema) -> graph_schema:
    state = schema.model_dump()
    post = state["post"]
    curated_post = llm_mistral.invoke(
        f"Curate the following linkedIn post in Gen-Z style :- \n{post}"
    ).content
    state["curated_post"] = curated_post
    state = graph_schema.validate(state)
    return state


from langgraph.graph import END, START, StateGraph

graph = StateGraph(graph_schema)

graph.add_node("create_post", create_post)
graph.add_node("curate_post", curate_post)

graph.add_edge(START, "create_post")
graph.add_edge("create_post", "curate_post")
graph.add_edge("curate_post", END)

pydantic_graph = graph.compile()


response = pydantic_graph.invoke(
    {
        "topic": "The importance of data privacy in digital age",
        "post": "",
        "curated_post": "",
    }
)
print(response)
