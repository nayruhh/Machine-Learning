# Embed out.pdf with OpenRouter (nvidia/nemotron-3.5-embed-1b:free)
# Requirements: pip install pypdf requests numpy
# Usage: set OPENROUTER_API_KEY first, then run: python embed_pdf.py

import os, sys, json, time
import requests
import numpy as np
from pypdf import PdfReader

# Settings
HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(HERE, "out.pdf")
MODEL = "nvidia/nemotron-3.5-embed-1b:free"
DIMS = 2048          # can truncate to 1024 / 512 / 256 to save memory on the device
CHUNK_WORDS = 300
OVERLAP_WORDS = 50
BATCH = 16

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("OPENROUTER_API_KEY is not set. Run:  set OPENROUTER_API_KEY=your-key")


def embed(texts, retries=6):
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": MODEL, "input": texts},
                timeout=60,
            )
            if r.status_code == 200:
                return [d["embedding"] for d in r.json()["data"]]
            print(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"Request error: {e}")
        wait = 2 ** attempt
        print(f"Retrying in {wait}s...")
        time.sleep(wait)
    sys.exit("Embedding failed after several retries. Try again in a few minutes.")


# 1. Extract text, keeping track of page numbers
reader = PdfReader(PDF_PATH)
words, pages = [], []
for pnum, page in enumerate(reader.pages, 1):
    w = (page.extract_text() or "").split()
    words += w
    pages += [pnum] * len(w)

print(f"Read {len(reader.pages)} pages, {len(words)} words")
if not words:
    sys.exit("No text found. The PDF is probably scanned images and needs OCR first.")

# 2. Chunk with overlap
step = CHUNK_WORDS - OVERLAP_WORDS
chunks = []
for i in range(0, len(words), step):
    chunks.append({
        "id": len(chunks),
        "text": " ".join(words[i:i + CHUNK_WORDS]),
        "pages": sorted(set(pages[i:i + CHUNK_WORDS])),
    })
    if i + CHUNK_WORDS >= len(words):
        break
print(f"Split into {len(chunks)} chunks")

# 3. Embed
vectors = []
for i in range(0, len(chunks), BATCH):
    vectors += embed([c["text"] for c in chunks[i:i + BATCH]])
    print(f"Embedded {len(vectors)}/{len(chunks)} chunks")

# 4. Truncate and normalize (cosine similarity becomes a simple dot product)
arr = np.array(vectors, dtype=np.float32)[:, :DIMS]
arr /= np.linalg.norm(arr, axis=1, keepdims=True)

# 5. Save next to the PDF
arr.tofile(os.path.join(HERE, "embeddings.bin"))
with open(os.path.join(HERE, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump({"model": MODEL, "dims": DIMS, "count": len(chunks),
               "chunks": chunks}, f, ensure_ascii=False, indent=1)

print(f"Done: {arr.shape[0]} chunks x {arr.shape[1]} dims")
print("Saved embeddings.bin and chunks.json")
