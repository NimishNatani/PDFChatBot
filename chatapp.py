import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("Hug_Face_API_Key")
groq_api_key = os.getenv("Groq_Api_Key")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("PDF Conversational Chatbot")
st.write("Upload PDF and chat with their content")

model = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)

# Get session_id from user
new_session_id = st.text_input("Session ID", value="default_session")
st.session_state.session_count = 0

# Reset session if session_id changes
if "session_id" not in st.session_state or st.session_state.session_id != new_session_id:
    if st.session_state.session_count>=2:
        st.warning("You have reached the maximum of 2 session")
        st.stop()
    else:
        st.session_state.session_id = new_session_id
        st.session_state.store = {}
        st.session_state.session_count += 1
        
        st.session_state.question_count = 0
        st.session_state.uploaded_files = None  # Reset the uploaded files
        st.write("Session reset. Please re-upload your PDFs.")

# Initialize session data
if "store" not in st.session_state:
    st.session_state.store = {}
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

# File uploader section
uploaded_files = st.file_uploader("Choose a PDF to upload", type="pdf", accept_multiple_files=False, key=st.session_state.session_id)

# If files are uploaded, process them
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files  # Save uploaded files in session
    documents = []
    for uploaded_file in uploaded_files:
        
        tempPdf = f'./{uploaded_files.name}temp.pdf'
        with open(tempPdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            fileName = uploaded_file.name

        loader = PyPDFLoader(tempPdf)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vector_store.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history. Do not answer the question,"
        "just reformulate it if needed and otherwise return it as is"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

    systemPrompt = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, say that you don't know the answer."
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", systemPrompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conservational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Display question count with circular progress
    max_questions = 5

    def render_circular_progress_bar(percentage,current_count,max_count):
        st.markdown(
             f"""
        <style>
        .progress-circle {{
            position: relative;
            width: 80px;
            height: 80px;
        }}

        .progress-circle svg {{
            transform: rotate(-90deg); /* Rotate for circular progress effect */
            width: 100%;
            height: 100%;
        }}

        .progress-circle circle {{
            fill: none;
            stroke-width: 8; /* Thickness of the circle */
        }}

        .progress-circle .bg {{
            stroke: #e0e0e0; /* Background circle color */
        }}

        .progress-circle .progress {{
            stroke: #4caf50; /* Green progress color */
            stroke-linecap: round; /* Rounded edges for progress */
            stroke-dasharray: 251.2; /* Circumference: 2πr (r = 40) */
            stroke-dashoffset: {251.2 - (251.2 * percentage / 100)}; /* Progress offset */
            transition: stroke-dashoffset 0.4s ease; /* Smooth animation */
        }}

        .progress-circle .progress-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%); /* Center the text */
            font-size: 1em;
            font-weight: bold;
            color: #4caf50;
        }}
        </style>
        <div class="progress-circle">
            <svg viewBox="0 0 100 100">
                <circle class="bg" cx="50" cy="50" r="40"></circle>
                <circle class="progress" cx="50" cy="50" r="40"></circle>
            </svg>
            <div class="progress-text">{current_count}/{max_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Create a layout with columns for user input and progress bar
    col1, col2 = st.columns([5, 1])

    # Render the user input in the left column
    with col1:
        user_input = st.text_input("Ask a question")
        if user_input:
            if st.session_state.question_count < max_questions:
                st.session_state.question_count += 1
                session_history = get_session_history(st.session_state.session_id)
                response = conservational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    })
                st.write("Assistant:", response['answer'])
            else:
                st.warning("You have reached the maximum of 5 questions for this session.")

    # Render the progress bar in the right column
    with col2:
        percentage = (st.session_state.question_count / max_questions)*100
        render_circular_progress_bar(percentage,st.session_state.question_count,max_questions)
