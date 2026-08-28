import langchain
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
# __import__('pprint').pprint(api_key)

# Simple LLM call with Streaming
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("mistral-small-latest", model_provider="mistralai")
# print(model)

messages = [
        SystemMessage("You are an expert CA studied from the best university of India and you are the best in the whole world, you believe in using examples for making people understand"),
        HumanMessage("What is new Tax Scheme. I have no idea about it. Explain how it is better or worse than the previous one")
        ]

# response = model.invoke(messages)
# print(response)

# Streaming example
# for chunk in model.stream(messages):
#     print(chunk.content, end = "", flush = True)

# Dynamic Prompt Templates
from langchain_core.prompts import ChatPromptTemplate

translate_template = ChatPromptTemplate.from_messages([
    ("system","You are a professional translator. Translate the following {text} from {source_lang} to {dest_lang} maintaining the tone and style"),
    ("user","{text}")
    ])

user_text = input("Enter Text to convert : ")
current_lang = input("Current Language : ")
destination_lang = input("Destination Language : ")

prompt = translate_template.invoke({
    "source_lang" : current_lang,
    "dest_lang" : destination_lang,
    "text" : user_text
    })

translated_text = model.invoke(prompt)
print(translated_text.content)


#Building First Chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
def create_story_chain():
    # Template for story generation
    story_prompt = ChatPromptTemplate.from_messages(
            [
                ("system","You are a creative story teller who plays with emotions and feelings. Write a short engaging story based on the given theme"),
                ("user","Theme : {theme}\nMain Character : {main_char}\nSetting : {setting}")
                ]
            )

    # Template for story analysis
    def analyze_story(story_text):
        return {"story":story_text}
    analysis_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a literary critic. Analyse the following story and provide insights"),
                ("user", "{story}")
                ]
            )
    story_chain = (
            story_prompt | model | StrOutputParser()
            )
    analysis_chain = (
            story_chain | RunnableLambda(analyze_story) | analysis_prompt | model | StrOutputParser()
            )
    return analysis_chain

chain = create_story_chain()
result = chain.invoke({
    "theme":"artificial intellegence",
    "main_char":"a curious robot",
    "setting":"a futuristic city"
    })
print(result)
