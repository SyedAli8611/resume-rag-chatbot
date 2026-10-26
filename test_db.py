import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

try:
    with psycopg.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), version();")
            print("Database connection successful!")
            print(cur.fetchone())

            cur.execute("""
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector';
            """)
            print("pgvector version:", cur.fetchone())

            cur.execute("""
                SELECT COUNT(*)
                FROM resume_chunks;
            """)
            print("Current resume chunks:", cur.fetchone()[0])

except Exception as exc:
    print("Database connection failed:", exc)