from dotenv import load_dotenv

load_dotenv()


from langchain.chat_models import init_chat_model

# -------------------------------- Task 1 - Prompt --------------------------------
from langchain_core.prompts import ChatPromptTemplate

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

# -------------------------------- Task 4 -  Chain Creation --------------------------------

chain = prompt_template | llm_mistral | str_parser

# -------------------------------- Task 5 -  Chain Invoke --------------------------------

response = chain.invoke({"topic": "What is the capital of INDIA?"})
print(response)

# -------------------------------- Task 6 -  Using Runnable for same --------------------------------

from langchain_core.runnables import RunnableSequence

chain1 = RunnableSequence(prompt_template, llm_mistral, str_parser)
response = chain1.invoke("Where is New Delhi located ?")
print(response)
