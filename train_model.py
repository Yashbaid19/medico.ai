import json
import requests
from pathlib import Path

# Paths
DATASET_DIR = Path("datasets")
TRAIN_FILE = DATASET_DIR / "train.json"
TEST_FILE = DATASET_DIR / "test.json"

# Ollama API URL
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "mistral"

# ---- Load dataset ----
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if not TRAIN_FILE.exists() or not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Dataset files not found in {DATASET_DIR}. "
        f"Make sure {TRAIN_FILE.name} and {TEST_FILE.name} exist."
    )

train_data = load_json(TRAIN_FILE)
test_data = load_json(TEST_FILE)

print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples.")

# ---- Detect keys dynamically ----
sample = train_data[0]
question_key = next((k for k in sample if "question" in k.lower()), None)

if not question_key:
    raise KeyError("Could not detect a 'question' key in dataset JSON.")

def get_answer(example):
    """Return the answer string, handling 'answer' (str) or 'answers' (list)."""
    if "answer" in example:
        return example["answer"]
    elif "answers" in example:
        ans = example["answers"]
        if isinstance(ans, list) and len(ans) > 0:
            return ans[0]
        return str(ans)
    else:
        return "[No answer found]"

def create_prompt(example):
    """Create a full instruction prompt."""
    return f"Patient question: {example[question_key]}\nProvide a helpful medical answer."

# ---- Call Ollama ----
def query_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[Error querying Ollama: {e}]"

# ---- Evaluate small subset ----
for i, ex in enumerate(test_data[:5], 1):  # only first 5 for speed
    prompt = create_prompt(ex)
    predicted = query_ollama(prompt)
    expected = get_answer(ex)

    print(f"\nSample {i}")
    print(f"Q: {ex[question_key]}")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
