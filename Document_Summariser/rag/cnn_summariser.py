from datasets import load_dataset
from document_loader import load_text_from_file
from utils import split_text_into_chunks
from embedder import Embedder
from retriever import VectorStore
from summarizer import Summarizer
import os

def summarize_cnn_articles(num_articles=3):
    print(f"\n📦 Downloading CNN/DailyMail articles...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{num_articles}]")

    os.makedirs("sample_docs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    embedder = Embedder()
    summarizer = Summarizer()

    for i, item in enumerate(dataset):
        article_text = item['article']
        article_file = f"sample_docs/cnn_article_{i+1}.txt"
        
        with open(article_file, "w") as f:
            f.write(article_text)

        print(f"\n🔹 Summarizing Article {i+1}")
        chunks = split_text_into_chunks(article_text)
        embeddings = embedder.embed(chunks)

        store = VectorStore(dim=len(embeddings[0]))
        store.add(embeddings, chunks)

        query_embedding = embedder.embed(["Summarize this document"])[0]
        top_chunks = store.query(query_embedding)
        final_input = " ".join(top_chunks[:3])

        summary = summarizer.summarize(final_input)
        print(summary)

        with open(f"results/summary_cnn_{i+1}.txt", "w") as f:
            f.write(summary)

if __name__ == "__main__":
    summarize_cnn_articles()
