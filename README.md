# RAG System for Business Support Consultation

## 📌 Project Description

This project is a Retrieval-Augmented Generation (RAG) system designed to automate the process of providing consultations on financial support measures for entrepreneurs.

The system is developed for the Chelyabinsk region “My Business” support center and combines:
- semantic document search using vector databases (FAISS),
- text chunking strategies,
- embedding models,
- and a generative LLM (GigaChat).

The goal is to provide fast and context-aware answers based strictly on internal regulatory documents.

---

## 🧠 System Architecture

The pipeline consists of the following stages:

1. **Document loading**
   - PDF parsing via `pdfplumber`

2. **Text chunking**
   - Recursive token-based chunking (LangChain + HuggingFace tokenizer)

3. **Embedding generation**
   - SentenceTransformer model (`bge-m3`)

4. **Vector storage**
   - FAISS IndexHNSW + ID mapping

5. **Retrieval**
   - Cosine similarity search

6. **Generation**
   - GigaChat LLM with RAG-enhanced prompt

---

## 🗂 Project Structure
.
├── api/
│   ├── main.py                 # FastAPI entry point
│   └── rag/
│       └── pipeline.py        # RAG pipeline orchestration (retrieval + LLM)

├── configs/
│   └── config.py              # Paths, model settings, constants

├── core/
│   ├── chunking/              # Text chunking strategies
│   ├── llm/                   # GigaChat integration and prompts
│   ├── retriever/             # Vector search (FAISS retrieval logic)
│   ├── document.py            # Document schema (metadata + text)
│   ├── loader.py              # PDF loading and preprocessing
│   └── vectorStore.py         # FAISS-based vector database

├── evaluation/
│   └── retrieval_evaluation.py # Hit Rate@K evaluation pipeline

├── frontend/
│   └── index.html            # Simple web UI (chat interface)

├── scripts/
│   ├── build_test_dataset.py  # Build dataset for evaluation
│   ├── build_vector_db.py     # Build FAISS index from documents
│   ├── debug_retrieval.py     # Debug retrieval results
│   ├── vec_db_summary.py      # Vector DB inspection tool
│   └── vector_db_builder.py   # Core pipeline for building index

└── README.md
