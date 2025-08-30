# main.py
import json
from pathlib import Path
import requests
import torch
from pdf_reader import load_pdfs_text
from semantic_search import build_embeddings, semantic_search

# ---- Configuration ----
PDF_FOLDER = Path("pdfs")              # folder containing PDFs
DATASET_FOLDER = Path("datasets")      # JSON dataset folder
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "mistral"
TOP_K = 3  # number of semantic search results to include in context

# ---- Ollama query function ----
def query_ollama(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[Error querying Ollama: {e}]"

# ---- Load PDFs ----
print("Loading PDFs...")
documents = load_pdfs_text(PDF_FOLDER)
print(f"Loaded {len(documents)} PDF documents.")

# ---- Build embeddings for PDFs ----
if documents:
    print("Building PDF embeddings...")
    pdf_embeddings, pdf_texts = build_embeddings(documents)
else:
    pdf_embeddings = torch.empty((0, 384))  # empty tensor with embedding size 384
    pdf_texts = []
print("PDF embeddings ready.")

# ---- Load dataset ----
dataset = []
train_file = DATASET_FOLDER / "train.json"
test_file  = DATASET_FOLDER / "test.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if train_file.exists() and test_file.exists():
    train_data = load_json(train_file)
    test_data = load_json(test_file)
    dataset = train_data + test_data
    print(f"Loaded {len(dataset)} QA pairs from dataset.")

# ---- Prepare dataset embeddings if dataset exists ----
if dataset:
    print("Building dataset embeddings...")
    qa_texts = [
        f"Patient question: {ex.get('question', '')}\nAnswer: {ex.get('answer', '')}" 
        for ex in dataset
    ]
    dataset_embeddings, _ = build_embeddings(qa_texts)

    # Merge PDF embeddings and dataset embeddings safely
    if pdf_embeddings.shape[0] == 0:
        embeddings = dataset_embeddings
    else:
        embeddings = torch.vstack([pdf_embeddings, dataset_embeddings])
    
    doc_texts = pdf_texts + qa_texts
    print("Dataset embeddings added.")
else:
    embeddings = pdf_embeddings
    doc_texts = pdf_texts

# ---- Chat loop ----
print("\nMedical chatbot ready! Type 'exit' to quit.\n")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot.")
        break

    # ---- Semantic search ----
    top_contexts = semantic_search(user_input, embeddings, doc_texts, top_k=TOP_K)
    context_text = "\n\n".join(top_contexts)

    # ---- Build final prompt ----
    final_prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {user_input}\nAnswer:"
    )

    # ---- Query Ollama ----
    answer = query_ollama(final_prompt)
    print(f"\nChatbot: {answer}\n")
