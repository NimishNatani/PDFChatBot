import streamlit as st
from pdf_processor import process_uploaded_pdf
from retriever_chain import build_rag_chain, get_session_history
from ui_components import render_circular_progress_bar
from langchain_community.chat_message_histories import ChatMessageHistory

# Streamlit Page Configuration
st.set_page_config(
    page_title="📘 PDF Conversational Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Configure your chatbot environment below:")

    new_session_id = st.text_input("🆔 Session ID", value="default_session")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload PDF File", type=["pdf"])
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Apply Theme ---

st.markdown("""
    <style>
    body { background-color: #0e1117; color: #fafafa; }
    .stChatMessage { background-color: #262730 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# --- Title & Intro ---
st.title("🤖 PDF Conversational Chatbot")
st.caption("Chat with your PDFs using **Groq + LangChain + HuggingFace**.")

# --- Session Handling ---
if "session_count" not in st.session_state:
    st.session_state.session_count = 0

if "session_id" not in st.session_state or st.session_state.session_id != new_session_id:
    if st.session_state.session_count >= 2:
        st.warning("⚠️ You’ve reached the maximum of 2 sessions.")
        st.stop()
    else:
        st.session_state.session_id = new_session_id
        st.session_state.store = {}
        st.session_state.session_count += 1
        st.session_state.question_count = 0
        st.session_state.uploaded_files = None
        st.info("🆕 Session reset. Please re-upload your PDFs.")

# --- Upload + Process PDF ---
if uploaded_file:
    documents = process_uploaded_pdf(uploaded_file)
    rag_chain = build_rag_chain(documents)
    st.session_state.uploaded_files = uploaded_file

    st.success(f"✅ {uploaded_file.name} processed successfully!")

    # --- Chat Area ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(msg["content"])

    # --- Input Area ---
    user_input = st.chat_input("💬 Ask a question about your PDF...")
    if user_input:
        if st.session_state.question_count < 5:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.question_count += 1

            with st.chat_message("assistant"):
                st.markdown("🧠 Thinking...")

            response = rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )

            answer = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.rerun()
        else:
            st.warning("⚠️ You’ve reached the maximum of 5 questions for this session.")

    # --- Sidebar Progress ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Progress")
    render_circular_progress_bar(st.session_state.question_count, 5)
else:
    st.info("📄 Upload a PDF from the sidebar to start chatting.")
