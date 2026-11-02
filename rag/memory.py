# Chat memory: saves every question and answer so you can look back at previous
# conversations and ask follow-up questions ("how much does it cost?").
# Used by both ask.py (naive RAG) and ask_crag.py (Corrective RAG).
#
# The history lives in its own file, chat_history.db, so rebuilding the knowledge base
# (ingest.py --rebuild) never deletes your conversations. Both backends and both
# systems share it; each saved answer records which system and model wrote it.
#
# How follow-up questions work:
#   1. The last few turns of the conversation are read from chat_history.db.
#   2. The LLM rewrites the follow-up into a stand-alone question
#      ("how much does it cost?" -> "How much does the Tesla Model Y battery pack cost?"),
#      because the search can only find chunks for words that are in the question.
#   3. The stand-alone question is searched as usual, and the conversation so far is
#      added to the answer prompt as background. The answer is still taken from the
#      sources only.

import re, json, sqlite3
from datetime import datetime

from rag_common import HERE, BACKEND, LLM_MODEL, chat

MEMORY_DB = HERE / "chat_history.db"
MEMORY_TURNS = 3         # previous question/answer pairs used for follow-ups
MAX_ANSWER_CHARS = 800   # long earlier answers are shortened in the prompt

REWRITE_PROMPT = """You rewrite follow-up questions for a document search system.
You receive the CONVERSATION SO FAR and a FOLLOW-UP question.
Rewrite the follow-up so it can be understood on its own: replace words like "it", "they",
"that", "the second one" with what they refer to, and add the topic if it is missing.
Keep the meaning and keep it short. If the follow-up is already clear on its own, or starts
a new topic, return it unchanged. Reply with the rewritten question only, no other text."""


# ---------- Storage ----------
def open_memory():
    db = sqlite3.connect(MEMORY_DB)
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            created TEXT,
            updated TEXT
        );
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY,
            session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
            question TEXT,        -- what the user typed
            standalone TEXT,      -- the question actually searched (after rewriting)
            answer TEXT,
            sources TEXT,         -- JSON list of "file | location" strings
            system TEXT,          -- "naive" or "crag"
            model TEXT,           -- backend and LLM that answered
            created TEXT
        );
    """)
    return db


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def get_session(mem, name, create=True):
    row = mem.execute("SELECT id FROM sessions WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    if not create:
        return None
    cur = mem.execute("INSERT INTO sessions (name, created, updated) VALUES (?, ?, ?)",
                      (name, now(), now()))
    mem.commit()
    return cur.lastrowid


def recent_turns(mem, session_id, n=MEMORY_TURNS):
    """The last n (question, answer) pairs, oldest first."""
    rows = mem.execute("SELECT standalone, answer FROM turns WHERE session_id = ? "
                       "ORDER BY id DESC LIMIT ?", (session_id, n)).fetchall()
    return rows[::-1]


def save_turn(mem, session_id, question, standalone, answer, sources, system):
    mem.execute("INSERT INTO turns (session_id, question, standalone, answer, sources, system, "
                "model, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, question, standalone, answer, json.dumps(sources), system,
                 f"{BACKEND}: {LLM_MODEL}", now()))
    mem.execute("UPDATE sessions SET updated = ? WHERE id = ?", (now(), session_id))
    mem.commit()


# ---------- Using the history ----------
def strip_citations(text):
    """Remove [Source N] labels: they refer to an earlier search, not the current one."""
    text = re.sub(r"\s*\[Sources? \d+(?:\s*(?:,|and|&)\s*(?:Source )?\d+)*\]", "", text)
    return text if len(text) <= MAX_ANSWER_CHARS else text[:MAX_ANSWER_CHARS] + "..."


def format_history(turns):
    return "\n\n".join(f"User: {q}\nAssistant: {strip_citations(a)}" for q, a in turns)


def rewrite_question(question, turns):
    """Turn a follow-up into a stand-alone question. No history means nothing to rewrite
    (and no extra LLM request)."""
    if not turns:
        return question
    messages = [{"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"CONVERSATION SO FAR:\n\n{format_history(turns)}"
                                            f"\n\nFOLLOW-UP: {question}"}]
    reply = chat(messages, temperature=0.0).strip().strip('"').strip()
    # Fall back to the original if the model replied with something that isn't a question
    if not reply or len(reply) > 4 * len(question) + 200 or "\n\n" in reply:
        return question
    return reply.splitlines()[0]


def add_history_to_prompt(messages, turns):
    """Put the conversation so far in front of the sources, clearly marked as background."""
    if turns:
        messages[1]["content"] = (
            "CONVERSATION SO FAR (background only: use it to understand the question, "
            "but take every fact from the SOURCES and cite only them):\n\n"
            f"{format_history(turns)}\n\n" + messages[1]["content"])
    return messages


# ---------- Commands for looking at previous chats ----------
def add_memory_args(parser):
    g = parser.add_argument_group("chat memory")
    g.add_argument("--chat", action="store_true",
                   help="interactive chat that remembers earlier questions")
    g.add_argument("--session", metavar="NAME",
                   help="conversation to continue (default for --chat: 'default')")
    g.add_argument("--sessions", action="store_true", help="list saved conversations")
    g.add_argument("--show", metavar="NAME", help="print a saved conversation")
    g.add_argument("--forget", metavar="NAME", help="delete a saved conversation")


def handle_memory_commands(args):
    """Run --sessions / --show / --forget. Returns True if one of them was given."""
    if not (args.sessions or args.show or args.forget):
        return False
    if not MEMORY_DB.exists():
        print("No saved conversations yet. Start one with --chat.")
        return True
    mem = open_memory()
    if args.sessions:
        list_sessions(mem)
    if args.show:
        show_session(mem, args.show)
    if args.forget:
        sid = get_session(mem, args.forget, create=False)
        if sid is None:
            print(f'No conversation called "{args.forget}".')
        else:
            mem.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            mem.commit()
            print(f'Deleted conversation "{args.forget}".')
    return True


def list_sessions(mem):
    rows = mem.execute("""
        SELECT s.name, COUNT(t.id), s.created, s.updated FROM sessions s
        LEFT JOIN turns t ON t.session_id = s.id GROUP BY s.id ORDER BY s.updated DESC
    """).fetchall()
    if not rows:
        print("No saved conversations yet. Start one with --chat.")
        return
    print(f"{'NAME':25} {'QUESTIONS':>9}   STARTED            LAST USED")
    for name, n, created, updated in rows:
        print(f"{name:25} {n:9}   {created}   {updated}")


def show_session(mem, name, last=None):
    sid = get_session(mem, name, create=False)
    if sid is None:
        print(f'No conversation called "{name}". See the list with --sessions.')
        return
    rows = mem.execute("SELECT question, standalone, answer, sources, system, model, created "
                       "FROM turns WHERE session_id = ? ORDER BY id", (sid,)).fetchall()
    if last:
        rows = rows[-last:]
    if not rows:
        print(f'Conversation "{name}" is empty.')
        return
    for i, (q, sq, a, src, system, model, created) in enumerate(rows, 1):
        print(f"--- {i}. {created} | {system} | {model} ---")
        print(f"You: {q}")
        if sq and sq != q:
            print(f"     (searched as: {sq})")
        print(f"\n{a}\n")
        sources = json.loads(src or "[]")
        if sources:
            print("Sources: " + "; ".join(sources) + "\n")


def run_chat(answer_fn, system, session_name):
    """Interactive loop. answer_fn(question, turns) -> (answer, sources, standalone)."""
    mem = open_memory()
    sid = get_session(mem, session_name)
    count = mem.execute("SELECT COUNT(*) FROM turns WHERE session_id = ?", (sid,)).fetchone()[0]
    print(f'Chat ({system} RAG, conversation "{session_name}", {count} earlier questions).')
    print("Commands: /history  show this conversation   /new  start fresh   /exit  quit\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if question.lower() == "/history":
            show_session(mem, session_name)
            continue
        if question.lower() == "/new":
            session_name = datetime.now().strftime("chat-%Y%m%d-%H%M%S")
            sid = get_session(mem, session_name)
            print(f'Started a new conversation "{session_name}".\n')
            continue
        turns = recent_turns(mem, sid)
        answer, sources, standalone = answer_fn(question, turns)
        save_turn(mem, sid, question, standalone, answer, sources, system)
        print()
    print(f'Saved. Continue later with:  --chat --session "{session_name}"')