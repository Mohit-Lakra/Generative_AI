# LLM Call
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

llm_mistral = ChatMistralAI(model="ministral-3b-latest", temperature=0.8)

# response = llm_mistral.invoke("Brother tell me a fun fact").content

# print(response)

from langchain.chat_models import init_chat_model

llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.7
)

response = llm_mistral.invoke(
    "Brother tell me a joke related to AI in current world"
).content
print(response)
