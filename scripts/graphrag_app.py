import os
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

sys.path.insert(0, SCRIPTS_DIR)
import importlib

hybrid = importlib.import_module("graphrag_04_hybrid_query")


CHART_COLORS = px.colors.qualitative.Set2


def render_auto_chart(sql_rows):
    """Try to turn this question's SQL result into a chart. If the shape of the
    data genuinely doesn't support one, say so plainly instead of staying silent."""
    if not sql_rows:
        return  # SQL wasn't used for this question at all -- nothing to say here,
                # the SQL evidence panel already covers that.

    columns, rows = sql_rows
    st.subheader("Dashboard")

    if not rows:
        st.info("The query returned no rows, so there's nothing to visualize.")
        return
    if len(columns) != 2:
        st.info(
            f"This result has {len(columns)} columns ({', '.join(columns)}) -- a chart "
            "needs exactly one category and one number, so this query isn't chart-ready."
        )
        return

    label_col, value_col = columns
    df = pd.DataFrame(rows, columns=columns)

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        st.info(
            f"'{value_col}' isn't a number, so this result doesn't map to a chart -- "
            "see the answer above instead."
        )
        return
    if len(df) < 2:
        st.info("Only one data point came back -- not enough to chart.")
        return

    truncated = False
    if len(df) > 15:
        df = df.sort_values(value_col, ascending=False).head(15)
        truncated = True

    looks_like_year = "year" in label_col.lower() or df[label_col].astype(str).str.match(r"^(19|20)\d{2}$").all()

    if looks_like_year:
        df = df.sort_values(label_col)
        fig = px.line(
            df, x=label_col, y=value_col, markers=True,
            title=f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}",
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_layout(template="plotly_white", margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = df.sort_values(value_col, ascending=True)
        col1, col2 = st.columns([3, 2])
        with col1:
            fig_bar = px.bar(
                df, x=value_col, y=label_col, orientation="h",
                title=f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}",
                color=label_col, color_discrete_sequence=CHART_COLORS, text=value_col,
            )
            fig_bar.update_layout(template="plotly_white", showlegend=False, margin=dict(t=50, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_pie = px.pie(
                df, names=label_col, values=value_col, hole=0.45,
                title="Share of total", color_discrete_sequence=CHART_COLORS,
            )
            fig_pie.update_layout(template="plotly_white", margin=dict(t=50, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

    if truncated:
        st.caption(f"Showing the top 15 of {len(rows)} categories.")

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
