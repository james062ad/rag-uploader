# 🔍 Retrieval-Augmented Generation (RAG) System with Comet Logging

This project is a complete end-to-end RAG (Retrieval-Augmented Generation) pipeline built with:

- ✅ FastAPI (backend API)
- ✅ PostgreSQL + pgvector (semantic vector DB)
- ✅ OpenAI (embedding + answer generation)
- ✅ Streamlit (frontend)
- ✅ Comet (Opik-hosted monitoring)

---

## ✨ Features

- Accept natural language questions via Streamlit or Swagger
- Search vector database for relevant chunks
- Use GPT to generate a grounded response
- Log all data to Comet: question, retrieved chunks, answer
- Iterative naming: RAG Query #1, #2, etc.

---

## 🛠️ Tech Stack

| Layer        | Tech                          |
|--------------|-------------------------------|
| Backend      | FastAPI + psycopg2            |
| Embedding    | OpenAI (`text-embedding-ada-002`) |
| Generation   | OpenAI GPT-3.5 Turbo          |
| Vector DB    | PostgreSQL + pgvector         |
| Frontend     | Streamlit                     |
| Monitoring   | Comet ML (Opik SDK)           |
| Secrets      | `.env` + `python-dotenv`      |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up `.env`

```env
OPENAI_API_KEY=sk-...
COMET_API_KEY=...
COMET_WORKSPACE=james1962
COMET_PROJECT_NAME=rag-project
```

### 3. Run FastAPI backend

```bash
uvicorn main:app --reload
```

### 4. Run Streamlit frontend

```bash
streamlit run app.py
```

---

## 📊 Comet Logging

Each query is tracked in your Comet dashboard:  
👉 https://www.comet.com/james1962/rag-project

### Logs include:

- 🧠 Question
- 📚 Retrieved Chunks
- 💬 Final Answer

Experiments are named like:
```
RAG Query #1
RAG Query #2
```

---

## 📂 Folder Structure

```
rag-project/
├── app.py               # Streamlit frontend
├── main.py              # FastAPI backend with Comet integration
├── create_table.py      # Creates DB table
├── insert_embedding.py  # Adds vectorized content to DB
├── retrieve_chunks.py   # Manual test of semantic search
├── requirements.txt
├── .gitignore
```

✅ `.env` and `venv/` are safely excluded from GitHub



## 🧠 Built by Geoffrey | March 2025
