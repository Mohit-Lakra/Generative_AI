import os
from dotenv import load_dotenv

load_dotenv()

if os.environ['OPENROUTER_API_KEY']:
    print("API Key is Set")

else :
    print("No API Key")

from langchain_openrouter import ChatOpenRouter

llm = ChatOpenRouter(model = "inclusionai/ling-3.0-flash-sante:free", temperature = 0)
# response = llm.invoke("What is AI ?")
# print(response.content)

# RAG implementation with our text data 

# #Creating Document
from langchain_core.documents import Document

my_text = """Natural language processing Natural language processing (NLP) allows programs to read, write, and communicate in human languages.[52] Specific problems include speech recognition, speech synthesis, machine translation, information extraction, information retrieval, and question answering.[53] Early work, based on Noam Chomsky's generative grammar and semantic networks, had difficulty with word-sense disambiguation[h] unless restricted to small domains called "micro-worlds" (due to the common sense knowledge problem[30]). British linguist and philosopher Margaret Masterman believed it was meaning and not grammar that was the key to understanding languages,[54] and that dictionaries and especially thesauri should be the basis of computational language structure.[55] Modern deep learning techniques for NLP include word embedding (representing words, typically as vectors encoding their meaning),[56] transformers (a deep learning architecture using an attention mechanism),[57] and others.[58] In 2019, generative pre-trained transformer (or "GPT") language models began to generate coherent text.[59][60] By 2023, these models were able to get human-level scores on the bar exam, SAT (Scholastic Assessment Test), GRE (Graduate Record Examination), and many other real-world applications.[61]
"""

docs = [Document(page_content = my_text, meta_data = {"source":"ABC", "documentID":"Doc1"})]

# #Splitting Document into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 30)
chunks = splitter.split_documents(docs)
print(len(chunks))


# #Creating Embeddings from Chunk
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="liquid/lfm-2.5-embedding-350m:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
)

vector = embedding_model.embed_query("What is artificial intelligence?")

print(len(vector))
print(vector[:5])


# #Create and Store embeddings in vector store
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
context = vectorstore.similarity_search("What is NLP ??", k = 2)

response = llm.invoke(f"What is NLP ?? You can answer using the following {context}")
print(response.content)

