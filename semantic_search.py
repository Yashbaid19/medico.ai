# semantic_search.py
from sentence_transformers import SentenceTransformer, util

# Load model once
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def build_embeddings(texts):
    """
    Returns embeddings tensor and the original texts.
    """
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings, texts

def semantic_search(query, embeddings, texts, top_k=3):
    """
    Returns top_k most similar texts to the query.
    """
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=top_k)
    top_texts = [texts[hit['corpus_id']] for hit in hits[0]]
    return top_texts
