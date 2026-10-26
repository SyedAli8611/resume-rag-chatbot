
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

# Project paths
PDF_PATH = Path(__file__).parent / "data" / "resume.pdf"
SOURCE = PDF_PATH.name

# Embedding model already used by the chatbot
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_text(text):
    """Split text into overlapping chunks, preferably at word boundaries."""
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Resume PDF not found: {PDF_PATH}"
        )

    print(f"Reading resume: {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        for chunk in split_text(text):
            chunks.append({
                "page_number": page_number,
                "chunk_text": chunk,
            })

    if not chunks:
        raise ValueError(
            "No text was extracted from the PDF. "
            "If the PDF is scanned, OCR may be required."
        )

    print(f"Extracted {len(chunks)} chunks.")
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = [item["chunk_text"] for item in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected 384-dimensional embeddings, "
            f"got {embeddings.shape[1]}"
        )

    print("Connecting to PostgreSQL...")

    with psycopg.connect(
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "Rag"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD"),
    ) as conn:
        register_vector(conn)

        # Replace previous chunks for this PDF to avoid duplicates.
        # The transaction rolls back if an error occurs.
        with conn.transaction():
            conn.execute(
                "DELETE FROM resume_chunks WHERE source = %s",
                (SOURCE,),
            )

            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO resume_chunks
                        (source, page_number, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (
                            SOURCE,
                            item["page_number"],
                            item["chunk_text"],
                            embedding,
                        )
                        for item, embedding in zip(chunks, embeddings)
                    ],
                )

        result = conn.execute(
            """
            SELECT COUNT(*)
            FROM resume_chunks
            WHERE source = %s
            """,
            (SOURCE,),
        ).fetchone()

    print("\nResume ingestion completed successfully!")
    print(f"Chunks saved to PostgreSQL: {result[0]}")
    print(f"Source PDF: {SOURCE}")


if __name__ == "__main__":
    main()
