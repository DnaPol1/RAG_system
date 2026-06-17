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
api/ # FastAPI application
core/ # Core modules (LLM, loader, vector store)
chunking/ # Text chunking strategies
retriever/ # Vector search logic
evaluation/ # Retrieval evaluation scripts
configs/ # Configuration files
scripts/ # Utility scripts (build DB, evaluation)
frontend/ # Web interface (HTML/CSS/JS)

