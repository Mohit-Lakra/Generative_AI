from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field

load_dotenv()


# ================================== CONDITIONAL CHAIN ==================================


# -------------------------------- Task 1 - Prompt --------------------------------

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a mood analyzer"),
        (
            "human",
            (
                """Analyze the mood of the following text: {text}.
            Return whether the mood is positive or negative."""
            ),
        ),
    ]
)


# -------------------------------- Task 2 - LLM --------------------------------

llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.3
)


# -------------------------------- Task 3 - Structured Output --------------------------------


class llm_schema(BaseModel):
    mood: str = Field(description="Mood must be either positive or negative")


llm_structured_output = llm_mistral.with_structured_output(llm_schema)


# -------------------------------- Task 4 - String Parser --------------------------------

str_parser = StrOutputParser()


# -------------------------------- Task 5 - Custom Runnable --------------------------------


def dictionary_maker(text: str) -> dict:
    return {"text": text}


dictionary_maker_runnable = RunnableLambda(dictionary_maker)


# -------------------------------- Task 6 - Conditional Chain - 1 --------------------------------


# --------------------- Task 6.1 - Prompt ---------------------

linkedin_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a LinkedIn post generator"),
        (
            "human",
            "Create an engaging LinkedIn post about DSA. "
            "Include relevant hashtags and a professional tone.",
        ),
    ]
)


# --------------------- Task 6.2 - LLM ---------------------

llm_mistral = init_chat_model(
    model="ministral-3b-latest", model_provider="mistralai", temperature=0.7
)


# --------------------- Task 6.3 - String Parser ---------------------

str_parser = StrOutputParser()

chain_linkedin = linkedin_prompt | llm_mistral | str_parser


# -------------------------------- Task 7 - Conditional Chain - 2 --------------------------------


def insta_chain(text: dict):

    # --------------------- Task 7.1 - Prompt ---------------------
    text = text["text"]

    insta_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an Instagram post generator"),
            (
                "human",
                """Create an engaging Instagram post about the movie {text}. 
                Include emojis, hashtags and an attractive caption.""",
            ),
        ]
    )

    # --------------------- Task 7.2 - LLM ---------------------

    llm_mistral = init_chat_model(
        model="ministral-3b-latest", model_provider="mistralai", temperature=0.7
    )

    # --------------------- Task 7.3 - String Parser ---------------------

    str_parser = StrOutputParser()

    chain_insta = insta_prompt | llm_mistral | str_parser

    result = chain_insta.invoke({"text": "KGF"})

    return result


insta_runnable = RunnableLambda(insta_chain)


# -------------------------------- Task 8 - Mood Extraction --------------------------------


def fetch_mood(result):
    print("Detected Mood:", result.mood)
    return result.mood.lower()


mood_fetch = RunnableLambda(fetch_mood)


# -------------------------------- Task 9 - Positive and Negative Keywords --------------------------------


positive_keywords = [
    "happy",
    "good",
    "great",
    "excited",
    "motivated",
    "awesome",
    "positive",
    "joyful",
    "confident",
    "energetic",
]

negative_keywords = [
    "sad",
    "bad",
    "angry",
    "upset",
    "depressed",
    "stressed",
    "negative",
    "tired",
    "frustrated",
    "disappointed",
]


def check_positive(mood):
    return any(word in mood for word in positive_keywords)


def check_negative(mood):
    return any(word in mood for word in negative_keywords)


# -------------------------------- Task 10 - Conditional Branch --------------------------------


conditional_chain = RunnableBranch(
    (check_positive, dictionary_maker_runnable | chain_linkedin),
    (check_negative, dictionary_maker_runnable | insta_runnable),
    dictionary_maker_runnable | insta_runnable,
)


# -------------------------------- Task 11 - Final Orchestration --------------------------------


final_orchestration = (
    prompt_template | llm_structured_output | mood_fetch | conditional_chain
)


# -------------------------------- Task 12 - Execution --------------------------------


print(
    final_orchestration.invoke({"text": "I am feeling very happy and motivated today!"})
)
