# ui.py
import json
import time
from pathlib import Path
import requests
import torch
import streamlit as st
from pdf_reader import load_pdfs_text
from semantic_search import build_embeddings, semantic_search

# ---- Page Configuration ----
st.set_page_config(
    page_title="Medical Chatbot with RAG",
    page_icon="🩺",
    layout="wide"
)

# ---- Configuration ----
PDF_FOLDER = Path("pdfs")
DATASET_FOLDER = Path("datasets")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "mistral"
TOP_K = 3

# Create folders if they don't exist
PDF_FOLDER.mkdir(exist_ok=True)
DATASET_FOLDER.mkdir(exist_ok=True)

# ---- Caching Functions for Performance ----

# Cache the expensive embedding model loading
# This is a resource, so it's not cleared by st.cache_data.clear()
@st.cache_resource
def get_embedding_model():
    """Loads the SentenceTransformer model and caches it."""
    from sentence_transformers import SentenceTransformer
    st.write("Loading embedding model... (this happens only once)")
    return SentenceTransformer("all-MiniLM-L6-v2")

# Cache the data loading and embedding generation
@st.cache_data
def load_and_embed_data():
    """
    Loads PDFs and JSON dataset, builds embeddings, and caches the result.
    """
    st.write("--- Starting Data Loading and Embedding ---")
    
    # --- Load PDFs ---
    st.write("1. Loading PDF texts from folder...")
    documents = load_pdfs_text(PDF_FOLDER)
    st.write(f"   -> Found {len(documents)} PDF documents.")
    
    if documents:
        st.write("2. Building PDF embeddings... (This can be slow for many/large PDFs)")
        pdf_embeddings, pdf_texts = build_embeddings(documents)
        st.write("   -> PDF embeddings created successfully.")
    else:
        st.write("2. No PDFs found, creating empty tensor.")
        pdf_embeddings = torch.empty((0, 384))
        pdf_texts = []
    
    # --- Load JSON Dataset ---
    dataset = []
    train_file = DATASET_FOLDER / "train.json"
    test_file = DATASET_FOLDER / "test.json"

    if train_file.exists() and test_file.exists():
        st.write("3. Loading JSON data...")
        def load_json(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        train_data = load_json(train_file)
        test_data = load_json(test_file)
        dataset = train_data + test_data
        st.write(f"   -> Loaded {len(dataset)} QA pairs from dataset.")
    else:
        st.write("3. JSON dataset not found.")

    # --- Prepare dataset embeddings ---
    if dataset:
        st.write("4. Preparing QA texts from JSON...")
        qa_texts = [
            f"Patient question: {ex.get('question', '')}\nAnswer: {ex.get('answer', '')}"
            for ex in dataset
        ]
        st.write("5. Building dataset embeddings... (This can be slow for large datasets)")
        dataset_embeddings, _ = build_embeddings(qa_texts)
        st.write("   -> Dataset embeddings created successfully.")
        
        device = dataset_embeddings.device 
        pdf_embeddings = pdf_embeddings.to(device) 
        
        st.write("6. Merging PDF and dataset embeddings...")
        embeddings = torch.vstack([pdf_embeddings, dataset_embeddings])
        doc_texts = pdf_texts + qa_texts
        st.write("   -> Embeddings merged.")
    else:
        st.write("6. No dataset found, using PDF embeddings only.")
        embeddings = pdf_embeddings
        doc_texts = pdf_texts
    
    st.success("--- Data loading and embeddings are ready! ---")
    return embeddings, doc_texts

# ---- Ollama query function ----
def query_ollama(prompt: str) -> str:
    """Sends a prompt to the Ollama API and returns the response."""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300) # 2 min timeout
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"[Error querying Ollama: {e}]"
    except Exception as e:
        return f"[An unexpected error occurred: {e}]"

# ---- Streamlit UI ----

st.title("🩺 Medical Chatbot with Document Upload")
st.markdown("This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded PDFs and a pre-existing medical Q&A dataset.")

# --- Sidebar for PDF Management ---
with st.sidebar:
    st.header("📄 Document Management")
    
    # PDF uploader
    uploaded_files = st.file_uploader(
        "Upload your PDF files here", 
        type="pdf", 
        accept_multiple_files=True
    )

    # THIS IS THE NEW, CORRECTED LOGIC
    if uploaded_files:
        files_saved = False
        for uploaded_file in uploaded_files:
            # Save the file to the PDF_FOLDER
            file_path = PDF_FOLDER / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            files_saved = True
        
        if files_saved:
            st.success(f"Uploaded {len(uploaded_files)} files.")
            st.info("Reloading data with new files...")
            # These two lines are the key to fixing the problem
            st.cache_data.clear()
            st.rerun()

    # THIS IS THE NEW MANUAL RELOAD BUTTON
    if st.button("🔄 Reload Data and Embeddings"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Current PDF Files in Folder:")
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    if pdf_files:
        for pdf in pdf_files:
            st.info(f"`{pdf.name}`")
    else:
        st.warning("No PDF files found in the 'pdfs' folder.")

# --- Load data and embeddings ---
embeddings, doc_texts = load_and_embed_data()

# --- Main Chat Interface ---
# (The rest of the code is unchanged)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            top_contexts = semantic_search(prompt, embeddings, doc_texts, top_k=TOP_K)
            context_text = "\n\n".join(top_contexts)

            final_prompt = (
                f"You are a helpful medical assistant. Use the following context to answer the question. "
                f"If the context is not sufficient, state that you cannot answer based on the provided documents.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {prompt}\n\nAnswer:"
            )

            response = query_ollama(final_prompt)
            
            with st.expander("🔍 View Context Used"):
                st.text(context_text)

            full_response = ""
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})