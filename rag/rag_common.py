# Shared settings and helpers for the multi-file RAG pipeline.
# Used by ingest.py (add documents), ask.py (plain RAG) and ask_crag.py (Corrective RAG).

import os, sys, time, sqlite3
from pathlib import Path

import numpy as np
import requests
import sqlite_vec
from dotenv import load_dotenv

load_dotenv()

# ---------- Settings ----------
# Where the models run:
#   "ollama"     = on this computer (offline, no daily limits, slower)
#   "openrouter" = OpenRouter's free online models (needs OPENROUTER_API_KEY)
# Can also be set for one terminal with:  $env:RAG_BACKEND = "openrouter"
BACKEND = os.environ.get("RAG_BACKEND", "ollama").lower()

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "documents"        # put all your files here (any topic, any mix)

if BACKEND == "ollama":
    EMBED_MODEL = "nomic-embed-text"     # install once with:  ollama pull nomic-embed-text
    LLM_MODEL = "gemma3:4b"
    DIMS = 768
    QUERY_PREFIX, DOC_PREFIX = "search_query: ", "search_document: "   # nomic-embed-text expects these
    DB_PATH = HERE / "knowledge_local.db"
elif BACKEND == "openrouter":
    EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b:free"
    LLM_MODEL = "nvidia/nemotron-3.5-lightning:free"   # any OpenRouter chat model works
    DIMS = 2048          # lower to 512 to save memory on the device (needs --rebuild)
    QUERY_PREFIX, DOC_PREFIX = "", ""
    DB_PATH = HERE / "knowledge.db"
else:
    sys.exit(f'Unknown RAG_BACKEND "{BACKEND}". Use "ollama" or "openrouter".')

OLLAMA_NUM_CTX = 8192    # how much text the local LLM can read at once (grading needs ~6,000 tokens)
CHUNK_WORDS = 300
OVERLAP_WORDS = 50
BATCH = 16               # chunks per embedding request
TOP_K = 4                # chunks given to the LLM
TEMPERATURE = 0.2

# Corrective RAG settings (used by ask_crag.py only; ask.py ignores them)
CANDIDATES = 10          # chunks retrieved and checked by the grader
MAX_CONTEXT = 6          # most chunks given to the LLM after grading
MIN_SIMILARITY = 0.05    # chunks below this are dropped before grading (the grader does the real filtering)
MAX_EXTRA_QUERIES = 4    # new search queries per question (unanswered parts + grader's suggestions)
EXTRA_PER_QUERY = 4      # chunks retrieved for each new query
PER_FILE = 2             # chunks per document, per query, for questions about several documents
MAX_GRADED = 16          # most chunks shown to the grader at once
GRADER_TEMPERATURE = 0.0 # grading should be consistent, not creative
# How the grader works: "per_chunk" asks a short yes/no question for each chunk (best for small
# local models), "batch" grades all chunks in one call (fewer requests for online models)
GRADING = "per_chunk" if BACKEND == "ollama" else "batch"
REFERENCE_THRESHOLD = 12 # chunks this dense with citations (years, DOIs, URLs) are reference lists

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
            elif r.status_code == 429 and "per-day" in r.text:
                # The daily free quota is used up; retrying within seconds can't help
                sys.exit("OpenRouter's daily limit for free models is used up "
                         "(50 requests per day, or 1,000 after buying 10 credits).\n"
                         "It resets at midnight UTC (03:00 Cairo time). Nothing was changed.")
            else:
                print(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"Request error: {e}")
        wait = 2 ** attempt
        print(f"Retrying in {wait}s...")
        time.sleep(wait)
    sys.exit("Request failed after several retries. Try again in a few minutes.")


# ---------- Ollama (local) ----------
def ollama_call(fn, **kwargs):
    """Call the local Ollama server with clear messages for the usual problems."""
    import ollama
    try:
        return getattr(ollama, fn)(**kwargs)
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            sys.exit(f"The model {kwargs.get('model')} is not installed. Run:  ollama pull {kwargs.get('model')}")
        sys.exit(f"Ollama error: {e}")
    except (ConnectionError, OSError) as e:
        sys.exit(f"Can't reach Ollama ({e}). Open the Ollama app (llama icon near the clock) and try again.")


# ---------- Embeddings and chat (same functions for both backends) ----------
def embed_texts(texts, is_query=False):
    """Embed a list of texts -> float32 array (len(texts), DIMS), normalized.
    Questions use is_query=True (some embedding models expect a different prefix)."""
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    out = []
    for i in range(0, len(texts), BATCH):
        batch = [prefix + t for t in texts[i:i + BATCH]]
        if BACKEND == "ollama":
            vectors = ollama_call("embed", model=EMBED_MODEL, input=batch)["embeddings"]
        else:
            data = post("embeddings", {"model": EMBED_MODEL, "input": batch})["data"]
            data.sort(key=lambda d: d.get("index", 0))
            vectors = [d["embedding"] for d in data]
        for v in vectors:
            v = np.array(v, dtype=np.float32)[:DIMS]
            out.append(v / (np.linalg.norm(v) or 1.0))
    return np.vstack(out)


def chat(messages, temperature=None, json_schema=None):
    """Send messages to the LLM and return its reply. json_schema forces a JSON reply
    of that shape (local models only; online models are asked for JSON in the prompt)."""
    temp = TEMPERATURE if temperature is None else temperature
    if BACKEND == "ollama":
        r = ollama_call("chat", model=LLM_MODEL, messages=messages, format=json_schema,
                        options={"temperature": temp, "num_ctx": OLLAMA_NUM_CTX})
        return r["message"]["content"]
    data = post("chat/completions", {"model": LLM_MODEL, "messages": messages,
                                     "temperature": temp})
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
        sys.exit(f"{DB_PATH.name} was built with {stored.get('embed_model')} at "
                 f"{stored.get('dims')} dims, but the settings say {EMBED_MODEL} at {DIMS}.\n"
                 "Run:  uv run rag/ingest.py --rebuild")
    return db


def to_blob(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()