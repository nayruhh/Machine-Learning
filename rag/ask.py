# Ask a question about all your documents: search knowledge.db for the most
# relevant chunks (from any file), then have the LLM answer using only those.
#
# Usage:
#   uv run rag/ask.py "your question"
#   uv run rag/ask.py "your question" --show-prompt   also print what is sent to the LLM
#   uv run rag/ask.py "your question" --search        only show the matching chunks (no LLM)
#   uv run rag/ask.py --list                          list the files in the store

import sys, argparse

from rag_common import DB_PATH, TOP_K, embed_texts, chat, open_db, to_blob

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
The sources are excerpts from different documents, which may cover different topics.
Rules:
- Use only the sources that are relevant to the question; ignore the others.
- Cite every fact with its source number, like [Source 2]. Use only these labels,
  never author names or references that appear inside the sources.
- If the sources do not contain the answer, say clearly that the documents do not cover it.
  Do not answer from your own knowledge.
- Be clear and concise."""


def search(db, question, k):
    q = embed_texts([question], is_query=True)[0]
    rows = db.execute("""
        WITH knn AS (
            SELECT rowid, distance FROM vec_chunks
            WHERE embedding MATCH ? AND k = ?
        )
        SELECT knn.distance, files.path, chunks.location, chunks.text
        FROM knn
        JOIN chunks ON chunks.id = knn.rowid
        JOIN files  ON files.id = chunks.file_id
        ORDER BY knn.distance
    """, (to_blob(q), k)).fetchall()
    # Vectors are normalized, so distance d relates to cosine similarity: cos = 1 - d^2 / 2
    return [(1 - d * d / 2, path, loc, text) for d, path, loc, text in rows]


def build_prompt(question, results):
    sources = "\n\n".join(
        f"[Source {i}] (file: {path}, {loc})\n{text}"
        for i, (_, path, loc, text) in enumerate(results, 1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"SOURCES:\n\n{sources}\n\nQUESTION: {question}"},
    ]


def list_files(db):
    rows = db.execute("SELECT path, file_type, chunks, added FROM files ORDER BY path").fetchall()
    if not rows:
        print("The store is empty. Add files to rag/documents and run: uv run rag/ingest.py")
        return
    for path, ftype, n, added in rows:
        print(f"{path:50} {ftype:5} {n:4} chunks   added {added}")
    print(f"\n{len(rows)} files, {sum(r[2] for r in rows)} chunks")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your documents")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--k", type=int, default=TOP_K, help="how many chunks to use")
    parser.add_argument("--search", action="store_true", help="only show matching chunks")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to the LLM")
    parser.add_argument("--list", action="store_true", help="list files in the store")
    args = parser.parse_args()

    if not DB_PATH.exists():
        sys.exit(f"{DB_PATH.name} not found. First run: uv run rag/ingest.py")
    db = open_db()

    if args.list:
        list_files(db)
        return
    if not args.question:
        sys.exit('Usage: uv run rag/ask.py "your question"')
    if db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
        sys.exit("The store is empty. Add files to rag/documents and run: uv run rag/ingest.py")

    question = " ".join(args.question)
    results = search(db, question, args.k)

    if args.search:
        for i, (sim, path, loc, text) in enumerate(results, 1):
            print(f"[{i}] {path} | {loc} | similarity {sim:.3f}")
            print(f"    {text[:300]}...\n")
        return

    messages = build_prompt(question, results)
    if args.show_prompt:
        print("=== Prompt sent to the LLM ===\n")
        for m in messages:
            print(f"--- {m['role']} ---\n{m['content']}\n")

    print("=== Answer ===\n")
    print(chat(messages))

    print("\n=== Sources used ===")
    for i, (sim, path, loc, _) in enumerate(results, 1):
        print(f"[Source {i}] {path} | {loc} | similarity {sim:.3f}")


if __name__ == "__main__":
    main()