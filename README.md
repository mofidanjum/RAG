# GraphRAG Agentic Data Analytics

A hands-on, step-by-step build of the full curriculum: **RAG → GraphRAG → GraphRAG + LLM → Agentic
Analytics**. Every stage is a real, working pipeline you run yourself, not a finished black box.

**Stage 1 (done):** plain RAG over a PDF — chunk, embed, store in Pinecone, retrieve, answer with Claude.
Deployed live: **https://rag-agenticanalytics.streamlit.app/**

**Stage 2 (done):** GraphRAG over patient records — the same data loaded three ways (SQLite, Neo4j,
Pinecone), queried together, to show exactly where plain RAG breaks and a graph doesn't.

**Stages 3–4 (next):** smarter routing between the three systems, then full agentic analytics.

---

## 0. One-time setup

```powershell
git clone https://github.com/mofidanjum/RAG.git "GraphRAG Agentic Data Analytics"
cd "GraphRAG Agentic Data Analytics"
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a `.env` file in the project root (never committed — already in `.gitignore`):

```
PINECONE_API_KEY=your-pinecone-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=your-neo4j-username
NEO4J_PASSWORD=your-neo4j-password
```

- Pinecone / Anthropic keys: needed for Stage 1 and Stage 2.
- Neo4j: only needed for Stage 2. Free instance at [neo4j.com/cloud/aura-free](https://neo4j.com/cloud/aura-free/) — after creating it, copy the URI/username/password from the connection details page (not the instance ID, which looks similar but isn't the username).

**Windows-only note:** if Neo4j connection scripts fail with `SSLCertVerificationError` or
`self-signed certificate in certificate chain`, that's Windows not having cached that certificate's
root CA yet — not a code or credentials problem. It's already worked around in every GraphRAG script
(they verify against `certifi`'s bundle instead of the OS store). If you hit it elsewhere, copy that
same pattern.

---

## Stage 1: Plain RAG (PDF → Pinecone → Claude)

Source document: `data/attention_is_all_you_need.pdf`. Run each script once, in order, the first time:

| Step | Command | What it does |
|---|---|---|
| 1 | `.\venv\Scripts\python.exe scripts\01_load_documents.py` | Loads the PDF, prints one sample page |
| 2 | `.\venv\Scripts\python.exe scripts\02_split_documents.py` | Splits pages into ~500-char chunks |
| 3 | `.\venv\Scripts\python.exe scripts\03_generate_embeddings.py` | Embeds sample chunks, prints the vector |
| 4 | `.\venv\Scripts\python.exe scripts\04_create_index.py` | Creates the Pinecone index (`attention-paper-rag`) |
| 5 | `.\venv\Scripts\python.exe scripts\05_upsert_embeddings.py` | Embeds every chunk, uploads to Pinecone |

Once steps 1–5 have run, only re-run these when you want to ask something:

| | Command | What it does |
|---|---|---|
| 6 | `.\venv\Scripts\python.exe scripts\06_query.py "your question"` | Retrieval only — top-3 chunks, no LLM |
| 7 | `.\venv\Scripts\python.exe scripts\07_rag_chain.py "your question"` | Full RAG — retrieves, then Claude answers |
| 8 | `.\venv\Scripts\python.exe -m streamlit run scripts\08_app.py` | Same thing, as a chat UI |

Test questions, including a deliberately unanswerable one: [`query.md`](query.md).

---

## Stage 2: GraphRAG (patient records → SQLite + Neo4j + Pinecone → Claude)

**The dataset**: 15 synthetic patients (real medical logic, fake people, via
[Synthea](https://synthetichealth.github.io/synthea/)) with rich visit histories, deliberately picked
so several patients share a diabetes/insulin history *and* have a completely unrelated condition
(like sinusitis) — the exact shape of question plain vector search struggles with, and a graph doesn't.

Run each script once, in order, to build all three systems from scratch:

| Step | Command | What it does |
|---|---|---|
| 1 | `.\venv\Scripts\python.exe scripts\build_patient_dataset.py` | Downloads/filters the Synthea data into `data/patients/` — 5 relational CSVs (patients, visits, conditions, medications, procedures) + one linked text note per visit |
| 2 | `.\venv\Scripts\python.exe scripts\graphrag_01_load_sqlite.py` | Loads the 5 CSVs into a real SQLite database (`data/patients/patients.db`) |
| 3 | `.\venv\Scripts\python.exe scripts\graphrag_02_load_graph.py` | Builds the Neo4j knowledge graph: `Patient → Visit → Condition/Medication/Procedure` |
| 4 | `.\venv\Scripts\python.exe scripts\graphrag_03_load_pinecone.py` | Embeds the visit notes into their own Pinecone index (`patient-notes-rag`) |

Once all four have run, ask questions against all three systems together:

| | Command | What it does |
|---|---|---|
| 5 | `.\venv\Scripts\python.exe scripts\graphrag_04_hybrid_query.py "your question"` | Claude writes a SQL query *and* a Cypher query, runs both plus a vector search, then answers using whichever evidence is actually useful |
| 6 | `.\venv\Scripts\python.exe -m streamlit run scripts\graphrag_app.py` | Same thing, as a UI — includes a "run pipeline" button (re-runs steps 1–4) and an "ask a question" box with expandable SQL / Cypher / vector evidence panels |

Try this question first — it's the one the whole dataset was built to demonstrate:

```
Which patients on insulin also have other unrelated conditions?
```

Then compare it against a plain-text search for the same question (no SQL, no graph) to see the
difference for yourself — SQL and Cypher both answer it completely; a raw vector search alone returns
near-duplicate insulin visits and misses the unrelated conditions entirely.

**Diagnostic script**: `scripts/graphrag_00_test_connection.py` — run this on its own if Neo4j
connection steps are failing, to confirm the certificate/credentials setup before touching anything else.

---

## Project structure

```
data/attention_is_all_you_need.pdf   Stage 1 source document
data/patients/                        Stage 2 dataset: 5 CSVs + patients.db + notes/
scripts/01-08_*.py                    Stage 1 pipeline, in order
scripts/build_patient_dataset.py      Stage 2: builds the dataset
scripts/graphrag_00_*.py              Stage 2: Neo4j connection diagnostic
scripts/graphrag_01-03_*.py           Stage 2: loads SQLite, Neo4j, Pinecone
scripts/graphrag_04_hybrid_query.py   Stage 2: the hybrid SQL+Cypher+vector query engine
scripts/graphrag_app.py               Stage 2: Streamlit UI
query.md                              Stage 1 test questions
requirements.txt                      pinned dependencies (both stages)
```

## Stack

- **Framework:** LangChain (`langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-anthropic`, `langchain-neo4j`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key)
- **Vector DB:** Pinecone (serverless, AWS us-east-1)
- **Relational DB:** SQLite (built into Python)
- **Graph DB:** Neo4j Aura (free tier)
- **LLM:** Anthropic Claude (`claude-sonnet-4-5-20250929`)
- **UI:** Streamlit

## Roadmap

- **Stage 3 — GraphRAG + LLM:** smarter routing (decide *which* of SQL/graph/vector actually apply to a
  question instead of always querying all three), and an explicit ontology layer on top of the graph
  (e.g. `Medication -[:TREATS]-> ConditionCategory`) so relationships carry real domain meaning, not
  just structure.
- **Stage 4 — Agentic analytics:** an agent that plans multi-step analysis across all systems rather
  than answering one question at a time.
