# 💼 Resume RAG Chatbot

A local, privacy-friendly interview assistant that answers questions about your resume using Retrieval-Augmented Generation (RAG).

Your resume PDF is split into chunks, embedded with a Sentence Transformers model, and stored in PostgreSQL with the **pgvector** extension. At question time, the most relevant chunks are retrieved by cosine similarity and passed to a local LLM served by **Ollama**, which answers using only your resume as context. Everything runs on your own machine.

## Features

- 📄 Ingests a resume PDF into a PostgreSQL + pgvector database
- 🔎 Semantic search over resume chunks (`all-MiniLM-L6-v2`, 384-dim embeddings)
- 🤖 Answers generated locally with Ollama (`qwen2.5:3b` by default)
- 💬 Streamlit chat UI with chat history and a "Resume passages used" panel showing sources and similarity scores
- 🛡️ Prompt grounded to resume content only — the model is told not to invent skills, dates or achievements

## How it works

```
data/resume.pdf
      │  ingest.py
      ▼
 PDF text ──► chunks (1000 chars, 150 overlap) ──► embeddings ──► PostgreSQL (pgvector)
                                                                        │
 User question ──► embedding ──► top-4 nearest chunks ◄─────────────────┘
                                        │
                                        ▼
                         Ollama (qwen2.5:3b) ──► answer in Streamlit
```

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit chat app: retrieves chunks from PostgreSQL and generates answers with Ollama |
| `ingest.py` | Reads `data/resume.pdf`, chunks and embeds it, and writes the chunks to PostgreSQL |
| `test_db.py` | Checks the database connection, the pgvector extension and the chunk count |
| `app_faiss_backup.py` | Earlier version of the app that used a FAISS index instead of PostgreSQL |
| `data/resume.pdf` | The resume used as the knowledge source |
| `.streamlit/config.toml` | Streamlit server settings |

## Prerequisites

- Python 3.10+
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.com/) running locally

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/SyedAli8611/resume-rag-chatbot.git
cd resume-rag-chatbot

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install "psycopg[binary]" pgvector python-dotenv
```

### 2. Pull the language model

```bash
ollama pull qwen2.5:3b
```

### 3. Create the database table

Connect to your database (default name `Rag`) and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS resume_chunks (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   VECTOR(384) NOT NULL
);
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
PGHOST=127.0.0.1
PGPORT=5432
PGDATABASE=Rag
PGUSER=postgres
PGPASSWORD=your_password_here
```

Verify the connection:

```bash
python test_db.py
```

### 5. Ingest your resume

Place your resume at `data/resume.pdf`, then run:

```bash
python ingest.py
```

Re-running ingestion replaces the existing chunks for that PDF, so it is safe to run again after updating your resume.

### 6. Start the chatbot

Make sure Ollama is running, then:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Example questions

- Tell me about yourself.
- Explain my projects.
- What are my MLOps skills?
- Explain my experience with Docker.

## Configuration

Key settings are constants at the top of `app.py` and `ingest.py`:

| Setting | Default | Where |
| --- | --- | --- |
| `LLM_MODEL` | `qwen2.5:3b` | `app.py` |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | `app.py` |
| `EMBEDDING_MODEL` / `MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | both |
| `TOP_K` | `4` | `app.py` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | `ingest.py` |

If you change the embedding model, update the `VECTOR(384)` dimension in the table and the dimension check in `ingest.py`.

## Troubleshooting

- **"Cannot connect to Ollama"** — start Ollama and confirm `ollama list` shows `qwen2.5:3b`.
- **"PGPASSWORD is missing"** — check that `.env` exists in the project root.
- **"No resume chunks found"** — run `python ingest.py`.
- **"No text was extracted from the PDF"** — the PDF is probably scanned; run OCR on it first.
