import os
import subprocess
import sys

import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

sys.path.insert(0, SCRIPTS_DIR)
import importlib

hybrid = importlib.import_module("graphrag_04_hybrid_query")

st.set_page_config(page_title="GraphRAG POC - Patient Records", layout="wide")
st.title("GraphRAG POC: Patient Records")

st.header("1. Build the pipeline")
st.write(
    "Runs the full data pipeline once: build the patient dataset, load it into SQLite, "
    "build the Neo4j graph, and embed the notes into Pinecone."
)

PIPELINE_STEPS = [
    ("Building patient dataset (CSVs + notes)", "build_patient_dataset.py"),
    ("Loading structured data into SQLite", "graphrag_01_load_sqlite.py"),
    ("Building the Neo4j knowledge graph", "graphrag_02_load_graph.py"),
    ("Embedding notes into Pinecone", "graphrag_03_load_pinecone.py"),
]

if st.button("Run pipeline"):
    for label, script in PIPELINE_STEPS:
        with st.status(label, expanded=True) as status:
            result = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS_DIR, script)],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            st.code(result.stdout or "(no output)")
            if result.returncode != 0:
                st.code(result.stderr)
                status.update(label=f"{label} - FAILED", state="error")
                st.error("Pipeline stopped due to the error above.")
                st.stop()
            status.update(label=f"{label} - done", state="complete")
    st.success("Pipeline complete. All three systems (SQLite, Neo4j, Pinecone) are loaded.")

st.divider()

st.header("2. Ask a question")
st.write("Your question is answered using SQL, the knowledge graph, and semantic search together.")


@st.cache_resource
def load_systems():
    db = hybrid.get_sql_database()
    graph = hybrid.get_graph()
    index = hybrid.get_pinecone_index()
    embedder = hybrid.get_embedder()
    llm = hybrid.get_llm()
    return db, graph, index, embedder, llm


question = st.text_input(
    "Question",
    value="Which patients on insulin also have other unrelated conditions?",
)

if st.button("Ask"):
    with st.spinner("Querying SQL, graph, and vector search, then asking Claude..."):
        db, graph, index, embedder, llm = load_systems()
        answer, evidence = hybrid.answer_question(question, db, graph, index, embedder, llm)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("SQL evidence"):
        st.code(evidence["sql_query"] or "(SQL not used for this question)")
        st.write(evidence["sql_result"])

    with st.expander("Graph (Cypher) evidence"):
        st.code(evidence["cypher_query"] or "(graph not used for this question)")
        st.write(evidence["cypher_result"])

    with st.expander("Vector search evidence"):
        for m in evidence["vector_matches"]:
            st.write(f"**{m['metadata']['patient_name']}** (score={m['score']:.4f})")
            st.text(m["metadata"]["text"])
