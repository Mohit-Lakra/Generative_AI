import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

if os.environ.get("Mistral_API_KEY") == False:
    raise ValueError("API Key not found")

llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.7
)

# -------------------------------- Messages --------------------------------

my_messages = [
    SystemMessage(
        content="You are my Gen-Z friend who always answers in a funny way and try to push me for the good"
    ),
    HumanMessage(content="Bro, Tell me a fun fact"),
]

# response= llm_mistral.invoke(my_messages).content
# print(response)

# -------------------------------- PROMPTS --------------------------------

from langchain_core.prompts import PromptTemplate

user_topic = input("Enter a topic for fun fact : ")
dynamic_prompt = PromptTemplate.from_template("Write a fun fact about {topic}")

ready_prompt = dynamic_prompt.invoke({"topic": user_topic})

# response = llm_mistral.invoke(ready_prompt).content
# print(response)


# -------------------------------- Setting Tone of AI --------------------------------

from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {tone} assistant"),
        ("user", "Write a fun fact about {topic}"),
    ]
)

ai_tone = input("Enter the tone of the AI : ")
ready_prompt = prompt_template.invoke({"topic": user_topic, "tone": ai_tone})
response = llm_mistral.invoke(ready_prompt).content
print(response)
