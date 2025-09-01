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

# ---- Inject Custom CSS for Animated Background ----
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css?family=Montserrat:400,400i,700");

:root {
  --speed-factor: 4;
  --img-1: url("data:image/png;base64,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");
  --img-2: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAXCAYAAABqBU3hAAAMEElEQVR4AQCBAH7/ANX////X////2v///97////h////4v///+L////h////3/7//934+f/c9PH/3PHr/97w6P/i8OX/5fHk/+ny5P/r8uP/7fLj/+7y4v/v8eP/8PLl//L06P/19uz/9/nx//r89f/6/fn/+fz6//b6+f/y9vf/7fHz/+nu8P/m6+7/AIEAfv8Azv3//9D+///T////1v///9n////b////2v7//9n5///X9Pb/1O7t/9Pq5v/U5+D/1ubc/9nm2v/d59n/4OjY/+Po1//k6Nf/5efX/+bn1//n6Nn/6enc/+zs4P/v7+X/8fHp//Hz7P/w8u7/7e/t/+nr6v/k5+f/4OPk/93h4v8AgQB+/wDC7f7/xO7//8bw///J8f//zPH9/83w+f/M7PL/yufq/8fh4f/F3Nj/xNfQ/8XUyv/H08b/ytPE/87Uw//R1cP/1NbC/9XVwf/W1cH/19TB/9jVw//Z1sX/3NnK/9/czv/h3tP/4d/W/+Df1//d3Nf/2djU/9TU0f/Q0M7/zc7M/wCBAH7/ALPZ5v+02ub/ttvm/7nc5f+62+L/u9nd/7nV1v+3z83/tMnE/7LEu/+xv7P/sbyu/7S7qv+3vKj/u72n/76+pv/Bvqb/wr6l/8O9pP/DvKX/xL2m/8a+qf/IwK3/y8Ox/83Gtv/Nx7n/zMa6/8nEuv/FwLf/wLu0/7u4sf+5ta//AIEAfv8AosPK/6PDyv+lxMn/p8TH/6jDxP+nwL7/pbu3/6O1rv+gr6T/namb/5yllP+do47/oKKL/6Ojif+npIj/q6WI/62lh/+upYb/r6SG/6+jhv+wo4f/saSJ/7Onjf+2qZL/uKyW/7mtmv+4rJv/tKqa/7CmmP+roZT/p52R/6Sbj/8AgQB+/wCTra7/lK2t/5WtrP+Wrar/lqum/5WnoP+Topj/kJyO/42Whf+KkHz/iox1/4uKcP+Oim3/kotr/5aNa/+ajmv/nI5q/52Naf+ejGj/noxo/56Maf+gjWv/oo9v/6SSdP+mlHj/p5V7/6aVff+iknz/no55/5mJdv+UhXP/koNx/wCBAH7/AIealP+HmpT/iJmS/4iYj/+Ilor/hpKE/4SMe/+AhnL/foBo/3x7YP98d1n/fnZV/4F2U/+GeFL/inpS/457Uv+QfFH/kXtQ/5F6T/+ReU//knlQ/5N6Uv+VfFb/l39a/5mBX/+agmL/mIFj/5V/Yv+Qel//i3Zc/4dyWf+Eb1f/AIEAfv8Afop//36Kfv9/inz/f4h5/36Fc/98gWz/eXtk/3Z1Wv9zb1H/cmpJ/3NoRP91Z0D/emg+/39qPv+DbT7/h24//4pvPv+Lbj3/i208/4tsPP+LbDz/jG0//45vQ/+QcUf/knRL/5J1Tv+RdE//jXFO/4hsS/+DZ0f/fmNE/3thQv8AgQB+/wB5f23/eX9t/3l+av95fGf/eHlh/3Z0Wv9zb1L/cGlJ/25kQP9tYDn/b140/3JeMf93YDD/fWMx/4JlMf+GZzL/iGgy/4lnMP+JZi//iWUv/4lkL/+KZTH/i2c1/45qOv+PbD7/j21A/45rQf+KaED/hGM8/35eOP96WTX/d1cy/wCBAH7/AHZ3Yf92dmD/dnVd/3VzWv90cFT/cmxN/3BnRf9uYT3/bF01/2xaLv9uWSr/c1oo/3hcKP9+Xyn/g2Iq/4hkKv+KZSr/i2Qo/4piJ/+JYSb/iWAn/4phKf+MYyz/jmYx/49nNf+PaDf/jWY4/4hjNv+CXTL/fFgt/3dTKf90UCf/AIEAfv8Ac3FX/3RwVv90b1T/c21Q/3JqS/9xZkT/b2I9/21dNf9sWS7/bVco/3BXJP91WCP/e1sj/4FeJP+GYSX/imMm/4xjJf+MYiP/jGEi/4tfIf+KXiH/i18j/41hJ/+OYyv/j2Uu/49lMf+MYzD/h14u/4FZKf96UiT/dE0g/3FKHf8AgQB+/wBxa0//cWtO/3FqTP9xaEn/cGZE/29iPv9tXjf/bFow/2xXKf9uVST/cVYi/3ZYIP99WyH/g14i/4hhI/+LYiP/jWIi/4xgIP+LXh3/ilwc/4lcHP+KXB7/i14i/41gJv+OYSn/jWEr/4pfKv+EWif/fVMi/3VMHP9vRhf/bEMU/wCBAH7/AGxlSf9sZUj/bGRG/21jQ/9sYT//bF45/2tbM/9rVy3/a1Un/25UI/9xVSD/dlcf/3xaH/+CXSD/h18g/4lgIP+KXx7/iV0b/4haGf+GWBf/hVcX/4ZYGf+HWR3/iVwh/4ldJP+IXCX/hFkk/35UIP92TBv/bkUU/2c+D/9jOwz/AIEAfv8AZV9D/2VfQ/9mXkH/Z10//2dcO/9nWjb/Z1cw/2dUK/9pUyb/a1Ii/29TH/90VR7/eVge/35aHv+CXB7/hFsc/4RaGv+CVxb/gFQT/35SEv9+URL/flEU/4BTGP+BVhz/glcf/4BWIP98Uh//dUwa/2xEFP9kPA3/XDUH/1gxA/8AgQB+/wBcWD//XVg+/15YPf9fWDv/YFc4/2FVNP9iUy//YlEq/2RQJf9nUCL/a1Ef/29SHv90VB3/eFYd/3tXG/98VRn/e1MV/3lPEf92TA7/dEoM/3RJDP90Sg//dkwT/3hPF/95URv/eFAc/3NMGv9rRRX/Yj0O/1k0B/9RLQD/TSgA/wCBAH7/AFNRPP9UUjv/VVI7/1dSOf9ZUjf/WlEz/1tQL/9dTyv/X04m/2JOI/9lTiD/aVAf/21RHf9xUhv/clEZ/3JPFf9wTBH/bUgN/2tECf9pQgj/aUIJ/2pEDP9tRxH/cEoW/3FMGv9vSxv/a0cZ/2NAFP9ZNwz/Ty4E/0cmAP9DIgD/AIEAfv8ASkw6/0tMOv9NTTr/T045/1FPN/9UTzX/Vk4x/1hNLf9aTSn/XU0m/2BNI/9kTiH/Z04f/2lOHP9pTBj/aEkU/2ZGD/9jQQv/YD4H/188Bv9fPQj/Yj8M/2VDEv9pSBj/a0oc/2pKHv9lRx3/XkAX/1Q3EP9JLQf/QSUA/zwgAP8AgQB+/wBCSDv/Q0k7/0VKO/9ITDv/S006/05OOP9RTjb/VE4y/1ZNL/9ZTSv/XE0o/19OJf9hTSL/Y0wf/2JKGv9gRhX/XUIQ/1o9C/9YOgj/VzkI/1k7Cv9cPxD/YUQX/2ZKHv9pTST/aU4m/2VLJf9dRCD/UzsY/0gxEP9AKQn/OyQE/wCBAH7/ADxHPv8+SD//QEk//0RLP/9ITT//S08+/05QPP9RUDn/VFA2/1dQMv9aUC//XE8r/15OKP9eTCT/XUkf/1tFGf9YQRP/VTwP/1M6DP9TOg3/VjwR/1tCF/9hSSD/Z1Ap/2tVMP9sVjP/aFQz/2FNLv9XRCb/TToe/0QyFv8/LRL/AIEAfv8AOEdD/zpIQ/89SkT/QU1F/0ZPRf9KUUX/TlND/1FUQf9UVD7/V1Q6/1pTN/9cUzP/XVEv/11PKv9bSyX/WEcf/1VCGf9SPhX/UTwT/1I9FP9WQRn/XEgi/2RQLP9sWTb/cV8+/3NhQ/9wYEP/aVo//19RN/9VRy//TD8n/0g6I/8AgQB+/wA3SEj/OEpI/zxMSv9AT0v/RVJM/0pVTP9OV0v/UlhI/1VYRv9YWEL/Wlg//1xXO/9dVTb/XFIx/1tOK/9YSiX/VEUg/1JBHP9RQBr/U0Id/1hHI/9gTyz/aVk4/3JiQ/94ak3/e21S/3lsU/9yZ0//aV5I/19VQP9WTDj/UUg0/wCBAH7/ADZKTP84S03/PE5O/0BSUP9FVVH/S1hR/09aUP9TW0//V1xM/1pcSf9cXEX/XVpB/15YPP9dVTf/W1Ex/1hNK/9VSCX/U0Ui/1NEIf9VRyT/W00r/2RWNf9uYEL/eGtO/39zWP+Cd1//gXdg/3tyXf9yaVb/aGBO/19YRv9aU0C/AYEAfv8ANktO/zhNT/88T1H/QFNT/0ZXVP9LWlT/UFxU/1ReUv9YXlD/W15M/11eSf9eXUT/X1o//15XOv9cUzT/WU4u/1ZKKP9URyX/VEYl/1dJKP9dUDD/Zlk7/3FlR/97cFT/g3hf/4d9Zv+FfWf/gHhk/3dwXf9tZlX/ZF5O/19aSv8gHKk1r6IVGAAAAABJRU5ErkJggg==');
}

html, body {
  width: 100vw;
  min-height: 100vh;
  font-family: Montserrat, sans-serif;
  margin: 0;
  padding: 0;
  display: flex;
  align-items: center;
  flex-direction: column;
  font-size: clamp(18px, 2.5vw, 30px);
  background-color: #e2efde;
  color: #262b29;
}

body:before, body:after {
  content: "";
  position: fixed;
  z-index: 0;
  inset: -100px;
  background-size: cover;
  pointer-events: none;
  aspect-ratio: 1;
}

body:before {
  mix-blend-mode: hard-light;
  animation: spin calc(1s * var(--speed-factor)) linear infinite;
  background-image: var(--img-2);
}

body:after {
  mix-blend-mode: lighten;
  animation: spin calc(2.5s * var(--speed-factor)) linear reverse;
  background-image: var(--img-3);
}

body {
  background-image: var(--img-1);
  background-size: cover;
  animation: hue 20s linear infinite;
}

@keyframes spin {
  100% {
    transform: rotate(360deg);
  }
}

@keyframes hue {
  0% {
    filter: Saturate(var(--saturate, 6)) Sepia(var(--sepia, 1)) hue-rotate(0deg);
  }
  100% {
    filter: Saturate(var(--saturate, 6)) Sepia(var(--sepia, 1)) hue-rotate(var(--hue, 360deg));
  }
}
</style>
""", unsafe_allow_html=True)

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