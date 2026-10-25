# Search the sqlite-vec store and print the matching chunks
# Requirements: pip install sqlite-vec numpy requests
# Usage: python search_store.py "your question here"
#        (OPENROUTER_API_KEY must be set in the same window)

import os, sys, json, sqlite3
import numpy as np
import requests
import sqlite_vec

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "vectors.db")
TOP_K = 3

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit('OPENROUTER_API_KEY is not set. In PowerShell run:  $env:OPENROUTER_API_KEY="your-key"')

question = " ".join(sys.argv[1:]) or input("Question: ")

db = sqlite3.connect(DB_PATH)
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

info = dict(db.execute("SELECT key, value FROM info").fetchall())
model, dims = info["model"], int(info["dims"])

# Embed the question with the SAME model, truncate and normalize
r = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": model, "input": [question]},
    timeout=60,
)
r.raise_for_status()
q = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)[:dims]
q /= np.linalg.norm(q)

# Nearest-neighbour search, then fetch the readable text
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
    [q.tobytes(), TOP_K],
).fetchall()

for cid, pages, text, dist in rows:
    similarity = 1 - dist ** 2 / 2   # cosine similarity for normalized vectors
    print(f"\n=== Chunk {cid} | pages {json.loads(pages)} | similarity {similarity:.3f} ===")
    print(text)

db.close()
