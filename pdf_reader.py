# pdf_reader.py
from pathlib import Path
import PyPDF2

def load_pdfs_text(folder: Path):
    """
    Reads all PDFs in the folder and returns a list of text strings.
    """
    documents = []
    for pdf_file in folder.glob("*.pdf"):
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            if text.strip():
                documents.append(text.strip())
    return documents
