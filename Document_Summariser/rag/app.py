from document_loader import load_text_from_file
from utils import split_text_into_chunks
from embedder import Embedder
from retriever import VectorStore
from summarizer import Summarizer
import os

def main(file_path):
    print(f"\n📄 Processing file: {file_path}")

    # Load and chunk
    text = load_text_from_file(file_path)
    chunks = split_text_into_chunks(text)

    # Embed chunks
    embedder = Embedder()
    embeddings = embedder.embed(chunks)

    # Store embeddings
    store = VectorStore(dim=len(embeddings[0]))
    store.add(embeddings, chunks)

    # Embed and retrieve
    query_embedding = embedder.embed(["Summarize this document"])[0]
    top_chunks = store.query(query_embedding)

    # Summarize
    summarizer = Summarizer()
    summary = summarizer.summarize(" ".join(top_chunks[:3]))

    print("\n✅ Summary:\n")
    print(summary)

    # Save summary
    os.makedirs("results", exist_ok=True)
    with open("results/summary.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    main("sample_docs/sample1.pdf")