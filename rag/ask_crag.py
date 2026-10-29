# Corrective RAG (CRAG): ask a question about your documents, but check the
# search results before answering, and fix the search when they are not good enough.
#
#   1. Retrieve:  get the CANDIDATES closest chunks (more than the 4 used by ask.py)
#   2. Grade:     an LLM "grader" marks each chunk relevant or not, says whether
#                 the relevant ones answer EVERY part of the question, and
#                 suggests new search queries for anything missing
#   3. Correct:   if something is missing, search again with those queries
#                 (and once per document for questions about several documents),
#                 then grade the new chunks
#   4. Answer:    the LLM answers from the relevant chunks only. If no chunk is
#                 relevant, the system says the documents don't cover it,
#                 without asking the LLM to answer at all.
#
# Usage:
#   uv run rag/ask_crag.py "your question"
#   uv run rag/ask_crag.py "your question" --trace        show every grading decision
#   uv run rag/ask_crag.py "your question" --show-prompt  print the prompt sent to the LLM
#
# ask.py (the original, naive RAG) is unchanged, so both can be compared on the same questions.

import sys, re, json, argparse

import numpy as np

from rag_common import (DB_PATH, CANDIDATES, MAX_CONTEXT, MIN_SIMILARITY, MAX_EXTRA_QUERIES,
                        EXTRA_PER_QUERY, PER_FILE, MAX_GRADED, GRADER_TEMPERATURE, GRADING,
                        REFERENCE_THRESHOLD, embed_texts, chat, open_db, to_blob)
from ask import build_prompt

NOT_COVERED = "The documents do not cover this question. No relevant passages were found."

GRADER_PROMPT = """You check search results for a document question-answering system.
You receive a QUESTION and numbered CANDIDATE excerpts from documents.

1. For each candidate, decide whether it states facts that answer the question, fully or
   partly. Reference lists, bibliography entries, citation titles and passages that only
   mention the same topic without the answer are NOT relevant.
2. Split the question into its separate parts (1 to 4 short parts). A question asking
   about two things, such as "how large are X and Y", has two parts. For each part,
   say whether the relevant candidates actually state the answer to it.
3. Write up to 3 short search queries that could find the unanswered parts in the
   documents, using the documents' likely wording.
4. Say whether the question asks about more than one document (for example
   "each document", "which documents", or a comparison between documents).

Reply with JSON only, no other text, in exactly this form:
{"relevant": [candidate numbers], "parts": [{"part": "...", "answered": true}], "queries": [], "multi_document": false}"""


# The exact shape the grader must reply in (enforced by Ollama for local models)
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "array", "items": {"type": "integer"}},
        "parts": {"type": "array", "items": {
            "type": "object",
            "properties": {"part": {"type": "string"}, "answered": {"type": "boolean"}},
            "required": ["part", "answered"]}},
        "queries": {"type": "array", "items": {"type": "string"}},
        "multi_document": {"type": "boolean"},
    },
    "required": ["relevant", "parts", "queries", "multi_document"],
}


# Per-chunk grading (small local models judge one passage at a time far more reliably)
RELEVANCE_PROMPT = """You decide whether ONE passage from a document helps answer a question.
Answer true only if the passage itself states facts that answer the question or part of it.
Answer false if the passage only mentions the same topic without giving the answer, or if it
is a reference list, bibliography or list of citations.
Reply with JSON only: {"relevant": true} or {"relevant": false}"""

RELEVANCE_SCHEMA = {"type": "object", "properties": {"relevant": {"type": "boolean"}},
                    "required": ["relevant"]}

SUFFICIENCY_PROMPT = """You check whether the passages found so far answer a question completely.
1. Split the question into its separate parts (1 to 4 short parts). A question asking
   about two things, such as "how large are X and Y", has two parts.
2. For each part, say whether the passages actually state the answer to it.
3. Write up to 3 short search queries that could find the unanswered parts in the
   documents, using the documents' likely wording.
4. Say whether the question asks about more than one document (for example
   "each document", "which documents", or a comparison between documents).
Reply with JSON only, in exactly this form:
{"parts": [{"part": "...", "answered": true}], "queries": [], "multi_document": false}"""

SUFFICIENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {"type": "array", "items": {
            "type": "object",
            "properties": {"part": {"type": "string"}, "answered": {"type": "boolean"}},
            "required": ["part", "answered"]}},
        "queries": {"type": "array", "items": {"type": "string"}},
        "multi_document": {"type": "boolean"},
    },
    "required": ["parts", "queries", "multi_document"],
}


def citation_density(text):
    """Citation markers per 100 words. Reference lists score far higher than normal text
    (measured on the test PDFs: reference chunks 13.7 or more, normal text 7.7 or less)."""
    words = max(len(text.split()), 1)
    years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", text))
    links = len(re.findall(r"doi|https?://|Available (?:online|at)|Accessed", text, re.I))
    pages = len(re.findall(r"\b\d+\s*\(\d+\)|\bpp?\.\s*\d+|\d+\s*[-\u2013]\s*\d+\.", text))
    return (years + 2 * links + pages) / words * 100


def read_json(reply):
    m = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------- Retrieval ----------
def knn(db, qvec, k):
    """Nearest chunks to one query vector, with their ids (so duplicates can be removed)."""
    rows = db.execute("""
        WITH knn AS (
            SELECT rowid, distance FROM vec_chunks
            WHERE embedding MATCH ? AND k = ?
        )
        SELECT chunks.id, knn.distance, files.path, chunks.location, chunks.text
        FROM knn
        JOIN chunks ON chunks.id = knn.rowid
        JOIN files  ON files.id = chunks.file_id
        ORDER BY knn.distance
    """, (to_blob(qvec), k)).fetchall()
    # normalized vectors: cosine similarity = 1 - distance^2 / 2
    return [{"id": cid, "sim": 1 - d * d / 2, "path": p, "loc": loc, "text": t, "via": "question"}
            for cid, d, p, loc, t in rows]


def similarity_to(db, qvec, chunk_id):
    """Similarity between the original question and a chunk found by another query."""
    blob = db.execute("SELECT embedding FROM vec_chunks WHERE rowid = ?", (chunk_id,)).fetchone()[0]
    return float(np.dot(qvec, np.frombuffer(blob, dtype=np.float32)))


def knn_per_file(db, qvec, n, skip=lambda c: False):
    """The n closest usable chunks from EACH document, so no document is crowded out.
    Chunks for which skip(chunk) is true (already seen, reference lists) don't count."""
    total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    best = {}
    for c in knn(db, qvec, min(total, 500)):
        group = best.setdefault(c["path"], [])
        if len(group) < n and not skip(c):
            group.append(c)
    return [c for group in best.values() for c in group]


class Log(list):
    """The list of steps shown to the user (also carries the question vector)."""
    qvec = None


# ---------- Grading ----------
def parse_parts(r, have_relevant):
    """Turn the grader's list of question parts into: complete?, what's missing, new queries.
    Unanswered parts are searched for in their own words first, then the grader's queries."""
    parts = [p for p in r.get("parts", []) if isinstance(p, dict) and str(p.get("part", "")).strip()]
    unanswered = [str(p["part"]).strip() for p in parts if not p.get("answered")]
    queries = []
    for q in unanswered + [q for q in r.get("queries", []) if isinstance(q, str)]:
        q = q.strip()
        if q and q.lower() not in (x.lower() for x in queries):
            queries.append(q)
    return {"sufficient": have_relevant and bool(parts) and not unanswered,
            "missing": "; ".join(unanswered),
            "queries": queries[:MAX_EXTRA_QUERIES],
            "multi_document": bool(r.get("multi_document", False))}


def parse_grade(reply, n):
    """Read the batch grader's JSON; return None if it can't be understood."""
    g = read_json(reply)
    if g is None:
        return None
    relevant = []
    for x in g.get("relevant", []):
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= x <= n and x not in relevant:
            relevant.append(x)
    return {"relevant": relevant, **parse_parts(g, bool(relevant))}


def grade(question, cands):
    """Decide which candidates are relevant and what is still missing.
    Reference lists are removed first without asking the LLM."""
    refs = [i for i, c in enumerate(cands, 1) if citation_density(c["text"]) >= REFERENCE_THRESHOLD]
    others = [(i, c) for i, c in enumerate(cands, 1) if i not in refs]
    if not others:
        g = grade_sufficiency(question, []) or {"sufficient": False, "missing": "", "queries": [],
                                                "multi_document": False}
        return {**g, "relevant": [], "refs": refs}
    if GRADING == "per_chunk":
        g = grade_per_chunk(question, others)
    else:
        sub = grade_batch(question, [c for _, c in others])
        g = {**sub, "relevant": [others[k - 1][0] for k in sub["relevant"]]}
    g["refs"] = refs
    return g


def grade_per_chunk(question, numbered):
    """One short yes/no call per chunk, then one call to check what is missing."""
    relevant, unreadable = [], 0
    for i, c in numbered:
        messages = [{"role": "system", "content": RELEVANCE_PROMPT},
                    {"role": "user", "content": f"QUESTION: {question}\n\n"
                                                f"PASSAGE (file: {c['path']}, {c['loc']}):\n{c['text']}"}]
        r = read_json(chat(messages, temperature=GRADER_TEMPERATURE, json_schema=RELEVANCE_SCHEMA))
        if r is None:
            unreadable += 1
            relevant.append(i)          # can't tell: keep it, the answer step can still ignore it
        elif bool(r.get("relevant")):
            relevant.append(i)
    kept = [c for i, c in numbered if i in relevant]
    g = grade_sufficiency(question, kept)
    if g is None:
        g = {"sufficient": bool(kept), "missing": "", "queries": [], "multi_document": False}
    return {**g, "relevant": relevant, "unreadable": unreadable}


def grade_sufficiency(question, kept):
    if kept:
        listing = "\n\n".join(f"[Passage {k}] (file: {c['path']}, {c['loc']})\n{c['text']}"
                               for k, c in enumerate(kept, 1))
    else:
        listing = "(none of the search results were relevant)"
    messages = [{"role": "system", "content": SUFFICIENCY_PROMPT},
                {"role": "user", "content": f"QUESTION: {question}\n\nPASSAGES:\n\n{listing}"}]
    r = read_json(chat(messages, temperature=GRADER_TEMPERATURE, json_schema=SUFFICIENCY_SCHEMA))
    return None if r is None else parse_parts(r, bool(kept))


def grade_batch(question, cands):
    """All candidates in one grading call (used for online models)."""
    listing = "\n\n".join(f"[Candidate {i}] (file: {c['path']}, {c['loc']})\n{c['text']}"
                          for i, c in enumerate(cands, 1))
    messages = [{"role": "system", "content": GRADER_PROMPT},
                {"role": "user", "content": f"QUESTION: {question}\n\nCANDIDATES:\n\n{listing}"}]
    g = parse_grade(chat(messages, temperature=GRADER_TEMPERATURE, json_schema=GRADE_SCHEMA), len(cands))
    if g is None:
        # Grader reply unreadable: fall back to plain RAG behaviour (top 4 by similarity)
        return {"relevant": list(range(1, min(4, len(cands)) + 1)), "sufficient": True,
                "missing": "", "queries": [], "multi_document": False, "failed": True}
    return g


def log_decisions(log, cands, g):
    if g.get("refs"):
        log.append(f"Removed {len(g['refs'])} reference-list chunks before grading")
    if g.get("unreadable"):
        log.append(f"{g['unreadable']} grader replies could not be read (those chunks were kept)")
    for i, c in enumerate(cands, 1):
        verdict = ("drop: reference list" if i in g.get("refs", []) else
                   "KEEP" if i in g["relevant"] else "drop")
        log.append(f"   [{verdict}] {c['path']} | {c['loc']} | similarity {c['sim']:.3f}")


# ---------- The corrective pipeline ----------
def corrective_retrieve(db, question, log):
    n_files = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    qvec = embed_texts([question], is_query=True)[0]
    log.qvec = qvec

    # 1. Retrieve wider than plain RAG, and drop chunks that are clearly unrelated
    cands = [c for c in knn(db, qvec, CANDIDATES) if c["sim"] >= MIN_SIMILARITY]
    log.append(f"Retrieved {len(cands)} candidates (similarity >= {MIN_SIMILARITY})")
    if not cands:
        return [], "NOT FOUND", ""

    # 2. Grade them
    g = grade(question, cands)
    if g.get("failed"):
        log.append("Grader reply could not be read; using the top 4 chunks (plain RAG)")
    relevant = [cands[i - 1] for i in g["relevant"]]
    log.append(f"Round 1: {len(relevant)} of {len(cands)} candidates relevant, "
               f"sufficient: {'yes' if g['sufficient'] else 'no'}")
    log_decisions(log, cands, g)

    # A question about several documents is only answered once every document was checked
    files_covered = {c["path"] for c in relevant}
    needs_all_docs = g["multi_document"] and len(files_covered) < n_files
    if needs_all_docs:
        log.append(f"Question covers several documents, but only {len(files_covered)} of "
                   f"{n_files} were found: searching each document")

    if g["sufficient"] and relevant and not needs_all_docs:
        return relevant, "CORRECT", g["missing"]

    # 3. Corrective retrieval: new queries from the grader, plus one search per document
    queries = g["queries"] or [question]
    if g["missing"]:
        log.append(f"Missing: {g['missing']}")
    log.append("New search queries: " + "; ".join(f'"{q}"' for q in queries))

    seen = {c["id"] for c in cands}
    new = []
    qvecs = embed_texts(queries, is_query=True)
    for qi, v in enumerate(qvecs, 1):
        for c in knn(db, v, EXTRA_PER_QUERY):
            if (c["id"] not in seen and c["sim"] >= MIN_SIMILARITY
                    and citation_density(c["text"]) < REFERENCE_THRESHOLD):
                c["via"] = f"new query {qi}"
                seen.add(c["id"]); new.append(c)
    if g["multi_document"]:
        # the closest chunks from each document, for the question and for every new query
        usable_seen = lambda c: c["id"] in seen or citation_density(c["text"]) >= REFERENCE_THRESHOLD
        for v in [qvec, *qvecs]:
            for c in knn_per_file(db, v, PER_FILE, skip=usable_seen):
                if c["id"] not in seen:
                    c["via"] = "per-document search"
                    seen.add(c["id"]); new.append(c)
    log.append(f"Corrective search found {len(new)} new candidates")
    if not new:
        return relevant, ("PARTIAL" if relevant else "NOT FOUND"), g["missing"]

    # Grade the kept chunks and the new ones together
    pool = (relevant + new)[:MAX_GRADED]
    g2 = grade(question, pool)
    if g2.get("failed"):
        log.append("Grader reply could not be read in round 2; using the top 4 chunks")
    relevant2 = [pool[i - 1] for i in g2["relevant"]]
    log.append(f"Round 2: {len(relevant2)} of {len(pool)} candidates relevant, "
               f"sufficient: {'yes' if g2['sufficient'] else 'no'}")
    log_decisions(log, pool, g2)

    if not relevant2:
        return [], "NOT FOUND", g2["missing"]
    return relevant2, ("CORRECTED" if g2["sufficient"] else "PARTIAL"), g2["missing"]


def main():
    parser = argparse.ArgumentParser(description="Ask questions with Corrective RAG")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--trace", action="store_true", help="show every grading decision")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to the LLM")
    args = parser.parse_args()

    if not DB_PATH.exists():
        sys.exit(f"{DB_PATH.name} not found. First run: uv run rag/ingest.py")
    if not args.question:
        sys.exit('Usage: uv run rag/ask_crag.py "your question"')
    db = open_db()
    question = " ".join(args.question)

    log = Log()
    chunks, status, missing = corrective_retrieve(db, question, log)

    print(f"=== Corrective step: {status} ===")
    for line in log:
        if args.trace or not line.startswith("   ["):
            print(line)

    if not chunks:
        print("\n=== Answer ===\n")
        print(NOT_COVERED)
        return

    # 4. Answer from the relevant chunks only, most similar to the question first.
    # Chunks found by a new query get their similarity to the ORIGINAL question,
    # so the numbers stay comparable with ask.py.
    for c in chunks:
        if c["via"] != "question":
            c["sim"] = similarity_to(db, log.qvec, c["id"])
    chunks = sorted(chunks, key=lambda c: -c["sim"])[:MAX_CONTEXT]
    results = [(c["sim"], c["path"], c["loc"], c["text"]) for c in chunks]
    messages = build_prompt(question, results)
    if status == "PARTIAL":
        gap = f" Likely missing: {missing}." if missing else ""
        messages[1]["content"] += ("\n\nNOTE: the search could not find everything." + gap +
                                   " Answer what the sources support and say clearly "
                                   "which parts the documents do not cover.")
    if args.show_prompt:
        print("\n=== Prompt sent to the LLM ===\n")
        for m in messages:
            print(f"--- {m['role']} ---\n{m['content']}\n")

    print("\n=== Answer ===\n")
    print(chat(messages))

    print("\n=== Sources used ===")
    for i, c in enumerate(chunks, 1):
        found = "" if c["via"] == "question" else f" | found by {c['via']}"
        print(f"[Source {i}] {c['path']} | {c['loc']} | similarity {c['sim']:.3f}{found}")


if __name__ == "__main__":
    main()