import os
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
DATA_DIR = os.path.join(BASE_DIR, "data", "patients")

sys.path.insert(0, SCRIPTS_DIR)
import importlib

hybrid = importlib.import_module("graphrag_04_hybrid_query")


def render_auto_chart(sql_rows):
    """If a SQL result looks like (category, number) pairs, chart it automatically."""
    if not sql_rows:
        return
    columns, rows = sql_rows
    if len(columns) != 2 or not rows:
        return

    df = pd.DataFrame(rows, columns=columns)
    numeric_col = columns[1]
    label_col = columns[0]
    if not pd.api.types.is_numeric_dtype(df[numeric_col]):
        return
    if not (1 < len(df) <= 20):
        return

    st.subheader("Auto-generated chart")
    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(df, x=label_col, y=numeric_col)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        fig_pie = px.pie(df, names=label_col, values=numeric_col)
        st.plotly_chart(fig_pie, use_container_width=True)

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

    render_auto_chart(evidence.get("sql_rows"))

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

st.divider()

st.header("3. Dashboards")
st.write(
    "Fixed views computed directly from the structured data (no LLM involved, so these are "
    "always reliable)."
)


@st.cache_data
def load_dashboard_data():
    patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
    visits = pd.read_csv(os.path.join(DATA_DIR, "visits.csv"))
    conditions = pd.read_csv(os.path.join(DATA_DIR, "conditions.csv"))
    medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
    return patients, visits, conditions, medications


try:
    patients_df, visits_df, conditions_df, medications_df = load_dashboard_data()

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.caption("Patient count by condition (top 8)")
        top_conditions = (
            conditions_df["DESCRIPTION"].value_counts().head(8).reset_index()
        )
        top_conditions.columns = ["Condition", "Count"]
        fig = px.bar(top_conditions, x="Count", y="Condition", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        st.caption("Patient count by race")
        race_counts = patients_df["RACE"].value_counts().reset_index()
        race_counts.columns = ["Race", "Count"]
        fig = px.pie(race_counts, names="Race", values="Count")
        st.plotly_chart(fig, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.caption("Patient count by city (region breakdown — all patients are in Massachusetts, so city is the meaningful regional split here)")
        city_counts = patients_df["CITY"].value_counts().reset_index()
        city_counts.columns = ["City", "Count"]
        fig = px.bar(city_counts, x="City", y="Count")
        st.plotly_chart(fig, use_container_width=True)

    with row2_col2:
        st.caption("Visit volume trend by year")
        visits_df["year"] = pd.to_datetime(visits_df["START"]).dt.year
        visits_by_year = visits_df.groupby("year").size().reset_index(name="Visits")
        fig = px.line(visits_by_year, x="year", y="Visits", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Top prescribed medications")
    top_meds = medications_df["DESCRIPTION"].value_counts().head(5).reset_index()
    top_meds.columns = ["Medication", "Prescriptions"]
    fig = px.bar(top_meds, x="Prescriptions", y="Medication", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.info("Run the pipeline (section 1) first to generate the dataset these dashboards read from.")
