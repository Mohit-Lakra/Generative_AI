from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# ================================== CHAIN WITH CUSTOM RUNNABLE ==================================

# -------------------------------- Task 1 - Prompt --------------------------------
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are an helpful Assistant"), ("human", "{topic}")]
)


# -------------------------------- Task 2 - LLM --------------------------------
llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.7
)

# -------------------------------- Task 3 - String Parser --------------------------------

from langchain_core.output_parsers import StrOutputParser

str_parser = StrOutputParser()

# -------------------------------- Task 4 - Custom Runnable --------------------------------
from langchain_core.runnables import RunnableLambda


def dictionary_maker(text: str) -> dict:
    return {"text": text}


dictionary_maker_runnable = RunnableLambda(dictionary_maker)


# -------------------------------- Task 5 - Template for Post --------------------------------

prompt_post = ChatPromptTemplate(
    messages=[
        ("system", "You are a social media post generator"),
        ("user", "Create a post for the following text for LinkedIn : {text}"),
    ]
)

# -------------------------------- Task 6 - Chain --------------------------------

chain = (
    prompt_template
    | llm_mistral
    | str_parser
    | dictionary_maker_runnable
    | prompt_post
    | llm_mistral
    | str_parser
)
response = chain.invoke("Today is solved 10 leetcode hard questions")
print(response)
