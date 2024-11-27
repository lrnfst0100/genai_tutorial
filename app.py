import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer

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
    embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    
    # Create FAISS VectorStore
    vectorstore = FAISS.from_texts(documents, embeddings)
    return vectorstore

# Function to create RAG pipeline
@st.cache_resource
def setup_rag_pipeline(vectorstore):
    # Load an open-source LLM from Hugging Face
    llm = pipeline("text2text-generation", model="google/flan-t5-base")

    # Define a wrapper for the pipeline to integrate with LangChain
    class HuggingFaceLLM:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def __call__(self, prompt):
            response = self.pipeline(prompt, max_length=512, truncation=True)
            return response[0]['generated_text']

    # Wrap the Hugging Face pipeline
    wrapped_llm = HuggingFaceLLM(llm)

    # Combine retriever and LLM in a RAG pipeline
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=wrapped_llm, retriever=retriever)
    return qa_chain

# Respond to user query
if query:
    vectorstore = setup_vector_db()
    qa_chain = setup_rag_pipeline(vectorstore)
    response = qa_chain.run(query)
    st.write("**Answer:**", response)
