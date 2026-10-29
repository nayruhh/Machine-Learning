# Document Q&A (RAG)

Ask questions about your own documents. The system splits each file into
overlapping chunks, embeds them, stores them in a local vector database
(`knowledge_local.db` for local models, `knowledge.db` for online ones), and when you ask a question it finds the most relevant chunks
and has an LLM answer **using only those chunks**, citing the file and location.

Files can be on any topic and in any mix of formats.

## How it works

```
documents/  ->  ingest.py  ->  knowledge.db  ->  ask.py  ->  answer + sources
 (your files)   (extract, chunk,   (sqlite-vec)    (search top chunks,
                 embed)                            prompt the LLM)
```

| File | Purpose |
|---|---|
| `ingest.py` | Reads every file in `documents/`, chunks and embeds it, saves it to the vector store |
| `ask.py` | Searches the store and has the LLM answer with cited sources (naive RAG) |
| `ask_crag.py` | Corrective RAG: grades the search results, searches again when something is missing, then answers |
| `rag_common.py` | All settings (models, chunk size, dimensions, ...) and shared helpers |
| `documents/` | Put your files here (not pushed to GitHub) |
| `knowledge_local.db` / `knowledge.db` | The vector store for local / online models, created by `ingest.py` (not pushed to GitHub) |

## Local or online models

`BACKEND` in `rag_common.py` chooses where the models run:

| Backend | Embedding model | LLM | Vector store | Notes |
|---|---|---|---|---|
| `ollama` (default) | nomic-embed-text (768 dims) | gemma3:4b | `knowledge_local.db` | Runs on this computer: offline, no daily limits, slower |
| `openrouter` | nvidia/nemotron-3.5-embed-1b (2048 dims) | nvidia/nemotron-3.5-lightning | `knowledge.db` | Free online models: 50 requests per day without credits |

Each backend keeps its own vector store, so switching doesn't need a rebuild.
For the local backend, install the models once:

```powershell
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

To use the online models for one terminal session: `$env:RAG_BACKEND = "openrouter"`.

## Corrective RAG (`ask_crag.py`)

`ask.py` gives the LLM the 4 closest chunks, whether or not they are useful.
`ask_crag.py` checks them first and fixes the search when they aren't good enough:

```
question -> retrieve 10 chunks -> GRADER: which are relevant? is every part answered?
              |                                      |
              |            yes ----------------------+---- no / partly
              |             |                                 |
              |             v                                 v
              |          CORRECT               search again with the grader's new
              |                                queries (+ each document separately
              |                                for questions about several documents)
              |                                               |
              |                                    grade the new chunks
              |                                               |
              |                         CORRECTED / PARTIAL / NOT FOUND
              v                                               v
         answer from the relevant chunks only     (NOT FOUND: "documents don't cover
                                                    this", without calling the LLM)
```

| Status | Meaning |
|---|---|
| CORRECT | The first search was good enough |
| CORRECTED | The first search missed something, and the corrective search found it |
| PARTIAL | Some relevant chunks were found, but part of the answer is still missing; the answer says which part |
| NOT FOUND | No relevant chunk at all; the system says the documents don't cover the question |

Cost: 2-3 LLM calls per question instead of 1 (one or two grading calls plus the answer),
so it is slower than `ask.py`. If the grader's reply can't be read, it falls back to plain RAG.

## Supported file types

| Type | Extensions | Location shown in citations |
|---|---|---|
| PDF | `.pdf` | page numbers |
| Word | `.docx` | section headings (or paragraph numbers if the file has no headings) |
| PowerPoint | `.pptx` | slide numbers (speaker notes included) |
| Text / Markdown | `.txt`, `.md` | line numbers |
| CSV | `.csv` | row numbers |
| Web pages | `.html`, `.htm` | line numbers |

Not supported: scanned PDFs (images of pages, need OCR), old `.doc` files,
Excel `.xlsx`, and images. These are skipped with a message.

## First-time setup

All commands are run in the VS Code terminal from the project folder
(`Machine-Learning`, the one containing `pyproject.toml`).

1. Install the libraries:

   ```powershell
   uv add pypdf requests numpy sqlite-vec python-dotenv python-docx python-pptx beautifulsoup4
   ```

2. Add your OpenRouter key to the `.env` file in the project folder
   (see `.env.example`):

   ```
   OPENROUTER_API_KEY=your-key-here
   ```

3. Create the documents folder (if it doesn't exist yet):

   ```powershell
   New-Item -ItemType Directory rag\documents
   ```

## Everyday use

**1. Add documents.** Copy files into `rag\documents` (subfolders are fine).

**2. Update the store:**

```powershell
uv run rag/ingest.py
```

Only new or changed files are embedded. Files you deleted from `documents`
are removed from the store. Run this again whenever you add, edit or delete files.

**3. Ask questions:**

```powershell
uv run rag/ask.py "What is the Consolidated Fund?"
```

## All commands

| Command | What it does |
|---|---|
| `uv run rag/ingest.py` | Add new files, re-embed changed files, remove deleted files |
| `uv run rag/ingest.py --rebuild` | Delete the vector store for the current backend and embed everything again |
| `uv run rag/ask.py "question"` | Answer a question from your documents, with sources |
| `uv run rag/ask.py "question" --k 6` | Give the LLM 6 chunks instead of 4 (for broader questions) |
| `uv run rag/ask.py "question" --search` | Only show the matching chunks and their similarity, no LLM |
| `uv run rag/ask.py "question" --show-prompt` | Also print the full prompt sent to the LLM |
| `uv run rag/ask.py --list` | List the files in the store and their chunk counts |
| `uv run rag/ask_crag.py "question"` | Answer with Corrective RAG (grades and corrects the search first) |
| `uv run rag/ask_crag.py "question" --trace` | Also show which chunks the grader kept and dropped in each round |
| `uv run rag/ask_crag.py "question" --show-prompt` | Also print the full prompt sent to the LLM |

## Example output

(Illustrative; your similarity numbers will differ.)

```
=== Answer ===

A complete sleep cycle lasts about 90 minutes [Source 1].

=== Sources used ===
[Source 1] human_sleep.docx | section: Why We Sleep; Stages of Sleep | similarity 0.412
[Source 2] human_sleep.docx | section: The Body Clock; How Much Sleep Do We Need? | similarity 0.301
[Source 3] climate_change.txt | lines 16-30 | similarity 0.122
[Source 4] out.pdf | pages 18-20 | similarity 0.098
```

The LLM is told to use only the relevant sources, so unrelated chunks
(like 3 and 4 above) are ignored. If no source contains the answer,
it says the documents don't cover it.

## Settings (`rag_common.py`)

| Setting | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `nvidia/nemotron-3.5-embed-1b:free` | Model that turns text into vectors |
| `LLM_MODEL` | `nvidia/nemotron-3.5-lightning:free` | Model that writes the answers (any OpenRouter chat model) |
| `DIMS` | `2048` | Vector size; use `512` to save memory on the embedded device |
| `CHUNK_WORDS` | `300` | Words per chunk |
| `OVERLAP_WORDS` | `50` | Words shared between neighbouring chunks |
| `BATCH` | `16` | Chunks per embedding request |
| `TOP_K` | `4` | Chunks given to the LLM per question |
| `TEMPERATURE` | `0.2` | Low = sticks closely to the sources |
| `CANDIDATES` | `10` | Corrective RAG: chunks retrieved and checked by the grader |
| `MAX_CONTEXT` | `6` | Corrective RAG: most chunks given to the LLM after grading |
| `MIN_SIMILARITY` | `0.05` | Corrective RAG: chunks below this are dropped before grading |
| `MAX_EXTRA_QUERIES` | `3` | Corrective RAG: new search queries the grader may suggest |
| `EXTRA_PER_QUERY` | `4` | Corrective RAG: chunks retrieved for each new query |
| `PER_FILE` | `2` | Corrective RAG: chunks per document for questions about several documents |
| `MAX_GRADED` | `16` | Corrective RAG: most chunks shown to the grader at once |
| `GRADER_TEMPERATURE` | `0.0` | Corrective RAG: grading should be consistent |

**After changing `EMBED_MODEL`, `DIMS`, `CHUNK_WORDS` or `OVERLAP_WORDS`,
run `uv run rag/ingest.py --rebuild`**, because questions must be embedded
the same way as the stored chunks. Changing `LLM_MODEL`, `TOP_K` or
`TEMPERATURE` takes effect immediately.

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv` is not recognized / `No pyproject.toml found` | You're in the wrong folder. `cd` into `Machine-Learning` (the folder with `pyproject.toml`) |
| `Can't reach Ollama` | Open the Ollama app (llama icon near the clock) |
| `The model ... is not installed` | Run the `ollama pull` command it shows |
| `OPENROUTER_API_KEY is not set` | Add the key to `.env` in the project folder |
| `HTTP 401` | The key is wrong or was deleted; create a new one at openrouter.ai |
| `Retrying in ...s` messages | The free models are busy; it retries automatically |
| `... was built with ... but the settings say ...` | You changed the model or `DIMS`; run `uv run rag/ingest.py --rebuild` |
| `skipped: almost no text found (scanned PDF?)` | The PDF is an image of pages; it needs OCR first |
| `failed to read this file` | The file is damaged or not really that type; re-save or re-export it |
| Yellow underlines on imports in VS Code | Select the `.venv` interpreter: Ctrl+Shift+P, Python: Select Interpreter |

## Notes

- Your API key lives only in `.env`, which is listed in `.gitignore`.
- `rag/documents/` and the `.db` files are also in `.gitignore`, so
  your documents are never pushed to GitHub. Anyone who clones the repo
  adds their own files and runs `ingest.py`.