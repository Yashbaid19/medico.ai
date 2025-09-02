🩺 Medico.ai

An intelligent toolkit for medical document processing and semantic search

🚀 Overview

Medico.ai is a Python-based application that empowers users to upload, process, and intelligently search medical documents (e.g., PDFs) via semantic embeddings and retrieval mechanisms. It packages essential ML workflows and a clean UI to streamline the management and retrieval of medical content.

✨ Features

📄 Document Ingestion – Parse and preprocess PDFs with ease

🔎 Semantic Search – Context-aware search using embeddings

🧠 Model Training Pipeline – Train, fine-tune, or leverage pretrained models

🎛️ Interactive Interface – Clean, user-friendly UI

📊 Colab Notebook – Accuracy/loss graphs and experiments

🗂️ Reusable Modules – Modular design (pdf_reader, semantic_search, etc.)

## 📸 Screenshots  

### 🔹 Application UI  
<div class="output">
  <h2>Output:</h2>
  <img src="https://github.com/user-attachments/assets/7c10224b-ceef-46ea-bbff-0fe1d92eb32b" alt="App UI Screenshot">
</div>





🔹 Accuracy & Loss Graphs
![WhatsApp Image 2025-08-30 at 12 13 27_02951ae9](https://github.com/user-attachments/assets/f4e32d1a-83a1-4afa-adf1-9812f2755598)
IPNYB notebook link : https://colab.research.google.com/drive/1RA3p9fGIxN7BVoqveRksCFmUKXxwkXaa?usp=sharing


🛠️ Getting Started
Prerequisites

Python 3.7+

pip & virtualenv

Installation
git clone https://github.com/Yashbaid19/medico.ai.git
cd medico.ai
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

▶️ Usage
1. Run the Application
python main.py

2. PDF Ingestion

Add your medical PDFs into the pdfs/ folder.
The ingestion workflow (in pdf_reader.py) handles text extraction and pre-processing.

3. Semantic Indexing & Search

Use semantic_search.py to generate embeddings and build a searchable vector index.

4. Model Training

Customize or train models with train_model.py.

5. Notebook (Training & Evaluation)

📂 Project Structure

<img width="939" height="343" alt="image" src="https://github.com/user-attachments/assets/1af6f786-2d9c-41ec-8e04-2abc1645b950" />


📦 Dependencies

(from requirements.txt, update with exact versions)

torch
transformers
faiss-cpu
flask
pandas
numpy

📈 Results

- **Achieved 93% accuracy on sample dataset**


Training/validation performance visualized in notebook graphs

End-to-end semantic search pipeline successfully demonstrated

💡 Best Practices

✅ Sanitize PDFs before ingestion for safety

✅ For large datasets, use FAISS or Pinecone vector DB

✅ Add logging & unit tests for reliability

✅ Extend modules for reusability

🤝 Contributing

Fork the repo

Create a new branch (git checkout -b feature/my-feature)

Implement your changes + tests

Commit (git commit -m "Add feature")

Push (git push origin feature/my-feature)

Open a Pull Request


🙌 Acknowledgements

Built with ❤️ using PyTorch, Hugging Face Transformers, and FAISS

Thanks to the open-source medical AI community for datasets & resources
