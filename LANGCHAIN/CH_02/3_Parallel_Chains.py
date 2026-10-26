from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()

# ================================== PARALLEL CHAIN ==================================

# -------------------------------- Task 1 - Prompt --------------------------------

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie summarizer"),
        ("human", "Please summarize the following movie in brief: {topic}"),
    ]
)


# -------------------------------- Task 2 - LLM --------------------------------
llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.7
)

# -------------------------------- Task 3 - String Parser --------------------------------


str_parser = StrOutputParser()

# -------------------------------- Task 4 - Custom Runnable --------------------------------


def dictionary_maker(text: str) -> dict:
    return {"text": text}


dictionary_maker_runnable = RunnableLambda(dictionary_maker)


# -------------------------------- Task 5 - Parallel Chain - 1 --------------------------------

# -------------------------------- Task 5.1 - Prompt --------------------------------

linkedin_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a linkedIn post generator"),
        ("human", "Create a post for the following text for linkedin : {text}"),
    ]
)

# -------------------------------- Task 5.2 - LLM --------------------------------

llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.7
)

# -------------------------------- Task 5.3 - String Parser --------------------------------

str_parser = StrOutputParser()
chain_linkedin = linkedin_prompt | llm_mistral | str_parser


# -------------------------------- Task 6 - Parallel Chain - 2 --------------------------------


def insta_chain(text: dict):

    text = text["text"]

    # --------------------- Task 6.1 - Prompt ---------------------
    insta_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a instagram post generator"),
            ("human", "Create a post for the following text for linkedin : {text}"),
        ]
    )

    # ----------------------- Task 6.2 - LLM -----------------------

    llm_mistral = init_chat_model(
        model="ministral-3b-latest", model_provider="mistralai", temperature=0.7
    )

    # --------------------- Task 6.3 - String Parser ---------------------

    str_parser = StrOutputParser()

    chain_insta = insta_prompt | llm_mistral | str_parser

    result = chain_insta.invoke(text)
    return result


insta_runnable = RunnableLambda(insta_chain)


# -------------------------------- Task 7 - Runnable Parallel --------------------------------

final_chain = (
    prompt_template
    | llm_mistral
    | str_parser
    | dictionary_maker_runnable
    | RunnableParallel(
        branches={"linkedin": chain_linkedin, "instagram": insta_runnable}
    )
)

# -------------------------------- Task 8 - Beautify --------------------------------


def beautify(final_response: dict) -> str:
    linkedin_response = final_response["branches"]["linkedin"]
    instagram_response = final_response["branches"]["instagram"]

    ans = "" + linkedin_response + "\n\n\n\n\n" + instagram_response
    return ans


beautify_runnable = RunnableLambda(beautify)


# -------------------------------- Task 9 - final_chain --------------------------------

final_beautified = final_chain | beautify_runnable
response = final_beautified.invoke({"topic": "KGF"})
print(response)
