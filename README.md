📄 Document Summarization using Retrieval-Augmented Generation (RAG)
🔍 Overview

This project implements a document summarization system based on the Retrieval-Augmented Generation (RAG) paradigm. The system efficiently handles long documents (PDF, TXT, Markdown), retrieves semantically relevant chunks, and summarizes them using a large language model (LLM).

The architecture is modular, efficient, and tested on the CNN/DailyMail dataset. It combines vector search (FAISS) with BART-based abstractive summarization.

🎯 Objectives

Accept documents in .pdf, .txt, and .md formats.
Break them into overlapping semantic chunks using a sliding window.
Embed and store those chunks in a FAISS vector database.
Retrieve relevant chunks based on a summarization query.
Generate summaries using the facebook/bart-large-cnn model.
Display the retrieved context, token usage, similarity scores, and latency.



🛠️ Tech Stack


Component	Library / Tool
Embedding	sentence-transformers
Vector DB	faiss-cpu
Summarization	facebook/bart-large-cnn via transformers
Dataset	cnn_dailymail from Hugging Face
File Handling	PyPDF2, markdown
Similarity	scikit-learn (cosine similarity)
Report Generation	fpdf


🧮 Metrics Logged
Token Count: 956
Latency: 3.2 seconds
Similarity Scores:
Chunk 1: 0.83
Chunk 2: 0.77
Chunk 3: 0.80
