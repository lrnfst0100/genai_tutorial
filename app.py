import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.llms import LLMChain
from langchain.runnables import Runnable

# Title
st.title("GenAI RAG App with Open-Source Models")
st.write("Ask a question based on the dataset!")

# Input for user query
query = st.text_input("Enter your question:")

# Function to setup vector database
@st.cache_resource
def setup_vector_db():
    # Load the dataset
    with open("data.txt", "r") as file:
        data = file.read()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = splitter.split_text(data)

    # Use SentenceTransformers for embeddings
    embeddings_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Just the model name as a string
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    # Create FAISS VectorStore
    vectorstore = FAISS.from_texts(documents, embeddings)
    return vectorstore

# Function to create RAG pipeline
@st.cache_resource
def setup_rag_pipeline(_vectorstore):
    # Load an open-source LLM from Hugging Face
    llm = pipeline("text2text-generation", model="google/flan-t5-base")

    # Define HuggingFaceLLM as a Runnable
class HuggingFaceLLM(Runnable):
    def __init__(self, model_name: str):
        self.pipeline = pipeline("text2text-generation", model=model_name)

    def invoke(self, prompt: str):
        response = self.pipeline(prompt, max_length=512, truncation=True)
        return response[0]['generated_text']

@st.cache_resource
def setup_rag_pipeline(_vectorstore):
    # Initialize LLM
    llm = HuggingFaceLLM(model_name="google/flan-t5-base")

    # Define retriever
    retriever = _vectorstore.as_retriever()

    # Setup QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    return qa_chain
    
# Respond to user query
if query:
    vectorstore = setup_vector_db()
    qa_chain = setup_rag_pipeline(vectorstore)
    response = qa_chain.run(query)
    st.write("**Answer:**", response)
