# RAG Demo — Attention Is All You Need

A hands-on Retrieval-Augmented Generation (RAG) pipeline built step by step: a PDF gets loaded,
split into chunks, embedded, stored in Pinecone, retrieved by semantic search, and answered by
Claude — wrapped in a Streamlit chat UI.

**Live app:** https://rag-agenticanalytics.streamlit.app/

## How it works

There are two separate flows that only meet at the final prompt — the whole document is never
"queried" directly; it's pre-processed into Pinecone once, and the question only ever searches
against that.

```
FLOW A — INDEXING (once, offline: scripts 01-05)

  PDF file
    │  (01) PyPDFLoader
    ▼
  Documents, one per page
    │  (02) RecursiveCharacterTextSplitter
    ▼
  Chunks (~500 chars each)
    │  (03) HuggingFaceEmbeddings (all-MiniLM-L6-v2, local, 384 dims)
    ▼
  Chunk vectors ──────(05) upsert──────▶  Pinecone index ("attention-paper-rag")
                                                    │
FLOW B — QUERYING (every time you ask: scripts 06/07/08)                │
                                                    │
  Your question (plain text)                        │
    │  (03) same embedder → question vector          │
    ▼                                                 │
  Semantic search ───────────────────────────────────┘
    │  (06) top-k most similar chunks come back, as text (not vectors)
    ▼
  Retrieved chunk texts  ─┐
                          ├──▶ (07) one prompt: "answer using ONLY this context" + chunks + your question
  Your question (plain text) ─┘         │
                                        ▼
                                  Claude generates an answer, grounded in the retrieved chunks
                                        │
                                        ▼ (08) Streamlit UI
                                  You, in a browser
```

The key point: your question's *embedding* is only ever used to search Pinecone. Once the top-k
chunks are found, everything that reaches Claude — the chunks and the question — is plain text
again, combined into a single prompt (see `RAG_PROMPT_TEMPLATE` in `scripts/07_rag_chain.py`).

## Setup

1. Clone the repo and create a virtual environment:
   ```powershell
   git clone https://github.com/mofidanjum/RAG.git
   cd RAG
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root (never committed — already gitignored) with:
   ```
   PINECONE_API_KEY=your-pinecone-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

## Running the pipeline step by step

Each script builds on the last. Run them in order the first time to see how the pipeline comes
together; after that, only `06_query.py` / `07_rag_chain.py` / `08_app.py` need re-running (01-05
just need to happen once, since the data is already stored in Pinecone).

| Script | What it does |
|---|---|
| `scripts/01_load_documents.py` | Loads the PDF, prints one sample page's text + metadata |
| `scripts/02_split_documents.py` | Splits pages into ~500-char chunks, prints a sample chunk |
| `scripts/03_generate_embeddings.py` | Embeds a couple of sample chunks, prints the vector |
| `scripts/04_create_index.py` | Creates the Pinecone index (`attention-paper-rag`, 384 dims, cosine) |
| `scripts/05_upsert_embeddings.py` | Embeds all chunks and upserts them into Pinecone |
| `scripts/06_query.py "<question>"` | Retrieval only — returns the top-3 most similar chunks, no LLM |
| `scripts/07_rag_chain.py "<question>"` | Full RAG — retrieves chunks, asks Claude to answer from them |
| `scripts/08_app.py` | Streamlit UI wrapping `07_rag_chain.py` |

Example:
```powershell
python scripts\01_load_documents.py
python scripts\02_split_documents.py
python scripts\03_generate_embeddings.py
python scripts\04_create_index.py
python scripts\05_upsert_embeddings.py
python scripts\06_query.py What is multi-head attention?
python scripts\07_rag_chain.py What is multi-head attention?
```

See [query.md](query.md) for a curated set of test questions, including one deliberately
out-of-scope question that tests whether Claude correctly says "not found in context" instead of
guessing.

## Running the app locally

```powershell
streamlit run scripts\08_app.py
```

Then open http://localhost:8501.

## Project structure

```
data/attention_is_all_you_need.pdf   the source document
scripts/01-08_*.py                    the pipeline, in order
query.md                              test questions for the RAG chain / app
requirements.txt                      pinned dependencies
```

## Stack

- **Framework:** LangChain (`langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-anthropic`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key)
- **Vector DB:** Pinecone (serverless, AWS us-east-1)
- **LLM:** Anthropic Claude (`claude-sonnet-4-5-20250929`)
- **UI:** Streamlit

## Roadmap

This is the "basic RAG" stage of a larger learning curriculum: RAG → GraphRAG → GraphRAG + LLM →
Agentic Analytics. Later stages will build on this pipeline.
