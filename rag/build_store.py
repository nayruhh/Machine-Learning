# Build a sqlite-vec vector store from embeddings.bin + chunks.json
# Requirements: pip install sqlite-vec numpy
# Usage: python build_store.py   (run after embed_pdf.py)

import os, json, sqlite3
import numpy as np
import sqlite_vec

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "vectors.db")

# Load the files produced by embed_pdf.py
meta = json.load(open(os.path.join(HERE, "chunks.json"), encoding="utf-8"))
vecs = np.fromfile(os.path.join(HERE, "embeddings.bin"), dtype=np.float32)
vecs = vecs.reshape(meta["count"], meta["dims"])

# Start fresh each time
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

db = sqlite3.connect(DB_PATH)
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# One table for the vectors, one normal table for the readable text
db.execute(f"CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{meta['dims']}])")
db.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT, pages TEXT)")
db.execute("CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT)")

with db:
    for c, v in zip(meta["chunks"], vecs):
        rowid = c["id"] + 1   # rowids start at 1
        db.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                   [rowid, v.astype(np.float32).tobytes()])
        db.execute("INSERT INTO chunks(id, text, pages) VALUES (?, ?, ?)",
                   [rowid, c["text"], json.dumps(c["pages"])])
    db.execute("INSERT INTO info VALUES ('model', ?)", [meta["model"]])
    db.execute("INSERT INTO info VALUES ('dims', ?)", [str(meta["dims"])])

print(f"Stored {meta['count']} chunks ({meta['dims']} dims) in vectors.db")
db.close()
