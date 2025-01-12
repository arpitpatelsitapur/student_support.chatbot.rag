import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Sidebar API Key Inputs
st.sidebar.subheader("API Keys Configuration")
gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
huggingface_api_key = st.sidebar.text_input("Enter your HuggingFace API Key", type="password")

# Warning if API keys are missing
if not gemini_api_key or not huggingface_api_key:
    st.sidebar.warning("Please provide both API keys to use the chatbot.")

# Define embeddings and LLM dynamically
embeddings = None
llm = None
if gemini_api_key and huggingface_api_key:
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=huggingface_api_key, model_name="hkunlp/instructor-xl")
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=gemini_api_key, temperature=0.6)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
vectordb_file_path = "faiss_index"


# Function to create vector DB
def create_vector_db():
    loader = CSVLoader(file_path='ggv_faqs.csv', source_column='Question')
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)

    loader1 = WebBaseLoader(["https://josaa.admissions.nic.in/applicant/seatmatrix/InstProfile.aspx?enc=nsU5HEkjvt/OC38zhsZ0ytGD/1D+L0n4WyLfOwyFk4="])
    data = loader1.load()
    docs1 = text_splitter.split_documents(data)
    vectordb.add_documents(docs1)
    
    vectordb.save_local(vectordb_file_path)
    st.success("Knowledge base created successfully!")


# Function to get QA chain
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """You are a helpful college office assistant who answers to the query of students regarding courses, fees, etc.
    Use the following pieces of context to answer the user's question. 
    If you don't know the answer, just say that you don't know; don't try to make up an answer.

    CONTEXT:{context}

    QUESTION:{question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


# Function to load directory to vector store
def load_directory_to_vector_store(dir_path):
    loader = DirectoryLoader(dir_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    vectordb.add_documents(texts)
    vectordb.save_local(vectordb_file_path)
    st.success("Directory loaded and knowledge base updated successfully!")


# Streamlit App Layout
st.title("Student HelpDesk")
st.markdown("Most of the information is specific to the Department of Information Technology, Guru Ghasidas University, Bilaspur")

# Sidebar Navigation
option = st.sidebar.radio(
    "Choose a section:",
    ("Home", "Chat with Chatbot", "Upload PDFs to Knowledge Base", "About")
)

if option == "Home":
    st.subheader("Welcome to the Student HelpDesk")
    st.write("""
        This platform provides a chatbot to assist students with their queries. 
        You can also create a knowledge base for the chatbot and explore additional features.
    """)
    st.image("Chatbot.png", caption="Student HelpDesk Demo", use_container_width=True)

elif option == "Chat with Chatbot":
    if not gemini_api_key or not huggingface_api_key:
        st.error("Both Gemini and HuggingFace API keys are required to use the chatbot.")
    else:
        st.subheader("Chat with the Chatbot")

        # Button to create knowledge base
        if st.button("Create Knowledge Base"):
            create_vector_db()

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        question = st.chat_input("Question: ")

        if question:
            st.chat_message('user').markdown(question)
            st.session_state.messages.append({'role': 'user', 'content': question})

            chain = get_qa_chain()
            response = chain.invoke(question)

            st.chat_message('assistant').markdown(response["result"])
            st.session_state.messages.append({'role': 'assistant', 'content': response["result"]})

elif option == "Upload PDFs to Knowledge Base":
    st.subheader("Upload PDFs to Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.read())
        load_directory_to_vector_store("./")

elif option == "About":
    st.subheader("About the HelpDesk")
    st.write("""
        The SoSE&T GGV Student HelpDesk is an AI-powered chatbot developed to provide instant responses to student queries. 
        The chatbot leverages LangChain's advanced QA chains and a custom knowledge base to deliver accurate and helpful answers.
        
        *Key Features:*
        - Streamlined access to department-specific information.
        - Interactive chatbot interface for real-time communication.
        - Ability to create and update knowledge bases.
    """)