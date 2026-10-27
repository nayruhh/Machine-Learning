# Document Q&A (RAG)

Ask questions about your own documents. The system splits each file into
overlapping chunks, embeds them, stores them in a local vector database
(`knowledge.db`), and when you ask a question it finds the most relevant chunks
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
| `ingest.py` | Reads every file in `documents/`, chunks and embeds it, saves it to `knowledge.db` |
| `ask.py` | Searches the store and has the LLM answer with cited sources |
| `rag_common.py` | All settings (models, chunk size, dimensions, ...) and shared helpers |
| `documents/` | Put your files here (not pushed to GitHub) |
| `knowledge.db` | The vector store, created by `ingest.py` (not pushed to GitHub) |

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
| `uv run rag/ingest.py --rebuild` | Delete `knowledge.db` and embed everything again |
| `uv run rag/ask.py "question"` | Answer a question from your documents, with sources |
| `uv run rag/ask.py "question" --k 6` | Give the LLM 6 chunks instead of 4 (for broader questions) |
| `uv run rag/ask.py "question" --search` | Only show the matching chunks and their similarity, no LLM |
| `uv run rag/ask.py "question" --show-prompt` | Also print the full prompt sent to the LLM |
| `uv run rag/ask.py --list` | List the files in the store and their chunk counts |

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

**After changing `EMBED_MODEL`, `DIMS`, `CHUNK_WORDS` or `OVERLAP_WORDS`,
run `uv run rag/ingest.py --rebuild`**, because questions must be embedded
the same way as the stored chunks. Changing `LLM_MODEL`, `TOP_K` or
`TEMPERATURE` takes effect immediately.

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv` is not recognized / `No pyproject.toml found` | You're in the wrong folder. `cd` into `Machine-Learning` (the folder with `pyproject.toml`) |
| `OPENROUTER_API_KEY is not set` | Add the key to `.env` in the project folder |
| `HTTP 401` | The key is wrong or was deleted; create a new one at openrouter.ai |
| `Retrying in ...s` messages | The free models are busy; it retries automatically |
| `knowledge.db was built with ... but the settings say ...` | You changed the model or `DIMS`; run `uv run rag/ingest.py --rebuild` |
| `skipped: almost no text found (scanned PDF?)` | The PDF is an image of pages; it needs OCR first |
| `failed to read this file` | The file is damaged or not really that type; re-save or re-export it |
| Yellow underlines on imports in VS Code | Select the `.venv` interpreter: Ctrl+Shift+P, Python: Select Interpreter |

## Notes

- Your API key lives only in `.env`, which is listed in `.gitignore`.
- `rag/documents/` and `rag/knowledge.db` are also in `.gitignore`, so
  your documents are never pushed to GitHub. Anyone who clones the repo
  adds their own files and runs `ingest.py`.