# -------------------------------- Guiding in Prompts --------------------------------
from dotenv import load_dotenv

load_dotenv()


from langchain.chat_models import init_chat_model

llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.7
)

# response = llm_mistral.invoke(
#     "Tell me a joke in a key value pair with the following keys :- Setup, PunchLine"
# ).content

# print(response)


# --------------------------------  Using Pydantic --------------------------------

from pydantic import BaseModel, Field


class llm_schema(BaseModel):
    setup: str = Field(description="Setup for the joke ")
    punchline: str = Field(description="Punch line for the joke")


llm_structured_output = llm_mistral.with_structured_output(llm_schema)
result = llm_structured_output.invoke("Tell me a joke")

print(result.setup)
print(result.punchline)
