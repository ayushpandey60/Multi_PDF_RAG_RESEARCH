import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Multi-PDF Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .main-title {
        font-family: 'Georgia', serif;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
      }
      .subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
      }
      .badge {
        display: inline-block;
        padding: 2px 10px;
        background: #eef2ff;
        color: #4338ca;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 6px;
      }
      .source-card {
        background: #f9fafb;
        border-left: 3px solid #6366f1;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "embeddings" not in st.session_state:
    with st.spinner("Loading embedding model (first run only)..."):
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory_messages" not in st.session_state:
    st.session_state.memory_messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Mistral API Key",
        value=os.getenv("MISTRAL_API_KEY", ""),
        type="password",
    )
    model_name = st.selectbox(
        "Mistral Model",
        ["mistral-large-latest", "mistral-small-latest", "open-mistral-7b"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    top_k = st.slider("Retriever Top-K", 1, 10, 4)
    enable_web = st.checkbox("🌐 Enable Web Search Fallback", value=True)

    st.divider()
    uploaded_files = st.file_uploader("Drop PDFs", type=["pdf"], accept_multiple_files=True)
    process_btn = st.button("⚙️ Process & Index PDFs", type="primary")

    if st.session_state.processed_files:
        st.success(f"✅ {len(st.session_state.processed_files)} file(s) indexed")

    if st.button("🗑️ Clear Chat & Index"):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.memory_messages = []
        st.session_state.processed_files = []
        st.rerun()

# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------
if process_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload a PDF.")
    else:
        with st.spinner("Processing..."):
            all_text = ""
            file_names = []
            for f in uploaded_files:
                reader = PdfReader(f)
                file_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                all_text += f"\n\n--- SOURCE: {f.name} ---\n\n" + file_text
                file_names.append(f.name)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(all_text)
            st.session_state.vectorstore = FAISS.from_texts(chunks, st.session_state.embeddings)
            st.session_state.processed_files = file_names
            st.session_state.num_chunks = len(chunks)
            st.rerun()

# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">📚 Multi-PDF Research Assistant</div>', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 Sources"):
                for s in msg["sources"]:
                    st.markdown(f'<div class="source-card">{s}</div>', unsafe_allow_html=True)

user_q = st.chat_input("Ask a question...")

if user_q:
    if not api_key:
        st.error("Add Mistral API key in sidebar.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_q, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        pdf_context, sources = "", []
        if st.session_state.vectorstore:
            docs = st.session_state.vectorstore.similarity_search(user_q, k=top_k)
            pdf_context = "\n".join([d.page_content for d in docs])
            sources = [f"<b>📄 Chunk:</b> {d.page_content[:100]}..." for d in docs]

        web_context = ""
        if enable_web and (st.session_state.vectorstore is None or any(w in user_q.lower() for w in ["latest", "news"])):
            web_context = DuckDuckGoSearchRun().run(user_q)
            if web_context: sources.append(f"<b>🌐 Web:</b> {web_context[:100]}...")

        # Updated line below: changed 'mistral_api_key' to 'api_key'
        llm = ChatMistralAI(model=model_name, temperature=temperature, api_key=api_key)
        
        messages = [SystemMessage(content="You are a research assistant. Use provided context.")]
        messages.extend(st.session_state.memory_messages[-10:])
        messages.append(HumanMessage(content=f"Context: {pdf_context}\nWeb: {web_context}\nQuestion: {user_q}"))

        response = llm.invoke(messages)
        st.markdown(response.content)

        st.session_state.chat_history.append({"role": "assistant", "content": response.content, "sources": sources})
        st.session_state.memory_messages.extend([HumanMessage(content=user_q), AIMessage(content=response.content)])