import os
from PyPDF2 import PdfReader
import markdown
import re

def load_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".txt":
        return open(filepath, "r", encoding="utf-8").read()
    elif ext == ".md":
        return markdown_to_text(filepath)
    else:
        raise ValueError("Unsupported file format.")

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)

def markdown_to_text(path):
    html = markdown.markdown(open(path, "r", encoding="utf-8").read())
    return re.sub('<[^<]+?>', '', html)
