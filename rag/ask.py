# Ask a question about the PDF: search vectors.db, then have an LLM answer
# using only the matching chunks.
# Requirements: pip install sqlite-vec numpy requests
# Usage: python ask.py "your question here"
#        (OPENROUTER_API_KEY must be set in the same window)

import os, sys, json, time, sqlite3
import numpy as np
import requests
import sqlite_vec

# Settings
HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "vectors.db")
LLM_MODEL = "nvidia/nemotron-3.5-lightning:free"   # any OpenRouter chat model works
TOP_K = 4               # how many chunks to give the LLM
SHOW_SOURCES = True     # print which chunks/pages were used

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit('OPENROUTER_API_KEY is not set. In PowerShell run:  $env:OPENROUTER_API_KEY="your-key"')

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def post(url, payload, retries=5):
    """POST to OpenRouter, retrying when the free models are busy."""
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"Request error: {e}")
        wait = 2 ** attempt
        print(f"Retrying in {wait}s...")
        time.sleep(wait)
    sys.exit("Request failed after several retries. Try again in a few minutes.")


# ---------- 1. Retrieve: find the chunks closest to the question ----------
def retrieve(question, k=TOP_K):
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    info = dict(db.execute("SELECT key, value FROM info").fetchall())
    embed_model, dims = info["model"], int(info["dims"])

    # Embed the question with the SAME model and dims as the stored chunks
    data = post("https://openrouter.ai/api/v1/embeddings",
                {"model": embed_model, "input": [question]})
    q = np.array(data["data"][0]["embedding"], dtype=np.float32)[:dims]
    q /= np.linalg.norm(q)

    rows = db.execute(
        """
        WITH matches AS (
            SELECT rowid, distance FROM vec_chunks
            WHERE embedding MATCH ? AND k = ?
        )
        SELECT chunks.id, chunks.pages, chunks.text, matches.distance
        FROM matches JOIN chunks ON chunks.id = matches.rowid
        ORDER BY matches.distance
        """,
        [q.tobytes(), k],
    ).fetchall()
    db.close()

    return [{"id": cid, "pages": json.loads(pages), "text": text,
             "similarity": 1 - dist ** 2 / 2}
            for cid, pages, text, dist in rows]


# ---------- 2. Augment: build the prompt from the chunks + question ----------
def build_messages(question, chunks):
    context = "\n\n".join(
        f"[Source {i} | pages {', '.join(map(str, c['pages']))}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    system = (
        "You answer questions about a document using ONLY the sources provided. "
        "If the sources do not contain the answer, say you could not find it in the document. "
        "Cite the sources you used like [Source 1]. Be concise."
    )
    user = f"Sources:\n\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# ---------- 3. Generate: have the LLM answer ----------
def generate(messages):
    data = post("https://openrouter.ai/api/v1/chat/completions",
                {"model": LLM_MODEL, "messages": messages, "temperature": 0.2})
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")

    chunks = retrieve(question)
    answer = generate(build_messages(question, chunks))

    print("\n=== Answer ===\n")
    print(answer)

    if SHOW_SOURCES:
        print("\n=== Sources used ===")
        for i, c in enumerate(chunks, 1):
            print(f"[Source {i}] chunk {c['id']} | pages {c['pages']} | similarity {c['similarity']:.3f}")
