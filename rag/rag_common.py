# Shared settings and helpers for the multi-file RAG pipeline.
# Used by ingest.py (add documents) and ask.py (ask questions).

import os, sys, time, sqlite3
from pathlib import Path

import numpy as np
import requests
import sqlite_vec
from dotenv import load_dotenv

load_dotenv()

# ---------- Settings ----------
HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "documents"        # put all your files here (any topic, any mix)
DB_PATH = HERE / "knowledge.db"      # the vector store

EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b:free"
LLM_MODEL = "nvidia/nemotron-3.5-lightning:free"   # any OpenRouter chat model works
DIMS = 2048              # lower to 512 to save memory on the device (needs --rebuild)
CHUNK_WORDS = 300
OVERLAP_WORDS = 50
BATCH = 16               # chunks per embedding request
TOP_K = 4                # chunks given to the LLM
TEMPERATURE = 0.2

OPENROUTER_URL = "https://openrouter.ai/api/v1"


# ---------- OpenRouter ----------
def _headers():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set. Add it to the .env file in the project folder.")
    return {"Authorization": f"Bearer {key}"}


def post(endpoint, payload, retries=6):
    """POST to OpenRouter, retrying when the free models are busy."""
    for attempt in range(retries):
        try:
            r = requests.post(f"{OPENROUTER_URL}/{endpoint}", headers=_headers(),
                              json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                if "error" not in data:
                    return data
                print(f"API error: {str(data['error'])[:200]}")
            elif r.status_code in (400, 401, 402, 403, 404):
                # These won't fix themselves by waiting (bad key, bad model name, ...)
                sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")
            else:
                print(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"Request error: {e}")
        wait = 2 ** attempt
        print(f"Retrying in {wait}s...")
        time.sleep(wait)
    sys.exit("Request failed after several retries. Try again in a few minutes.")


def embed_texts(texts):
    """Embed a list of texts -> float32 array (len(texts), DIMS), normalized."""
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        data = post("embeddings", {"model": EMBED_MODEL, "input": batch})["data"]
        data.sort(key=lambda d: d.get("index", 0))
        for d in data:
            v = np.array(d["embedding"], dtype=np.float32)[:DIMS]
            out.append(v / (np.linalg.norm(v) or 1.0))
    return np.vstack(out)


def chat(messages):
    data = post("chat/completions", {"model": LLM_MODEL, "messages": messages,
                                     "temperature": TEMPERATURE})
    return data["choices"][0]["message"]["content"]


# ---------- Database ----------
def open_db():
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(f"""
        CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,         -- relative to the documents folder
            hash TEXT,                -- fingerprint, to detect changed files
            file_type TEXT,
            chunks INTEGER,
            added TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,   -- same number as the rowid in vec_chunks
            file_id INTEGER REFERENCES files(id),
            location TEXT,            -- e.g. "pages 3-4", "slide 2", "lines 10-40"
            text TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{DIMS}]);
    """)
    # The stored vectors must use the same model and size as new questions.
    stored = dict(db.execute("SELECT key, value FROM info").fetchall())
    if not stored:
        db.executemany("INSERT INTO info VALUES (?, ?)",
                       [("embed_model", EMBED_MODEL), ("dims", str(DIMS))])
        db.commit()
    elif stored.get("embed_model") != EMBED_MODEL or stored.get("dims") != str(DIMS):
        sys.exit(f"knowledge.db was built with {stored.get('embed_model')} at "
                 f"{stored.get('dims')} dims, but the settings say {EMBED_MODEL} at {DIMS}.\n"
                 "Run:  uv run rag/ingest.py --rebuild")
    return db


def to_blob(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()