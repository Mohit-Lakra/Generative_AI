import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os

st.set_page_config(page_title = "Welcome to Fatafat AI", page_icon = "🗿")

st.title("🗿 Welcome to Fatafat AI")
st.markdown("Learning Basics of Langchain, So created a chatbot")

with st.sidebar:
    st.header("Settings")

    # API Key
    api_key = st.text_input("API Key of MISTRAL : ", type = "password")

    # Model Selection
    model_name = st.selectbox(
            "Model",
            ["mistral-small-latest","mistral-medium-latest","mistral-large-latest"]
            )

    # Clear Button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM
@st.cache_resource
def get_chain(Api_key,model_name):
    if not Api_key:
        return None
    # Initialize model
    model = init_chat_model(model_name, api_key=Api_key, model_provider="mistralai", temperature = 0.7, streaming = True)

    # Initialize prompt templates
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a helpul assistant powered by MISTRAL, answer questions clearly and use example where ever possible and always try to connect with user emotionally like some fun or serious words use emojis to represent emotions "),
        ("human","{question}")
        ])

    # Create Chain
    chain = prompt | model | StrOutputParser()

    return chain

chain = get_chain(api_key,model_name)
if not chain:
    st.warning("❌ Wrong API Key. Check it again")

else :
    # Display Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input 
    if question := st.chat_input("Ask me Anything "):
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in chain.stream({"question":question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response+" ")
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role":"assistant", "content":full_response})
            except Exception as e:
                st.error(f"Error : {str(e)}")

st.markdown("---")
st.markdown("### 💡 Try these examples : ")
col1,col2 = st.columns(2)

with col1:
    st.markdown("- What is Langchain? ")
    st.markdown("- What is LPU technology? ")

with col2:
    st.markdown("- Compare new vs old tax regime of India ")
    st.markdown("- Write a haiku about use of AI in Web Devlopement")

st.markdown("---")
st.markdown("Built with Langchain and Mistral 😁")
