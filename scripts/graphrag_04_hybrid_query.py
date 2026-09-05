import os
import sys

from dotenv import load_dotenv

load_dotenv()

import certifi
from langchain_anthropic import ChatAnthropic
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from neo4j import TrustCustomCAs
from pinecone import Pinecone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_PATH = os.path.join(BASE_DIR, "data", "patients", "patients.db")
PINECONE_INDEX_NAME = "patient-notes-rag"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

SQL_PROMPT = """You are a SQLite expert. Given this schema:

{schema}

Write ONE SQLite query that helps answer this question: "{question}"

Return ONLY the raw SQL query - no explanation, no markdown fences, no semicolon.
If SQL genuinely cannot help answer this question, return exactly: NONE"""

CYPHER_PROMPT = """You are a Neo4j Cypher expert. Given this graph schema:

{schema}

Write ONE Cypher query that helps answer this question: "{question}"

Return ONLY the raw Cypher query - no explanation, no markdown fences.
If the graph genuinely cannot help answer this question, return exactly: NONE"""

SYNTHESIS_PROMPT = """Answer the question using ONLY the evidence below, gathered from three
different systems. Say which system(s) supported your answer. If the evidence doesn't
contain the answer, say so plainly instead of guessing.

--- SQL database results ---
{sql_evidence}

--- Knowledge graph results ---
{graph_evidence}

--- Semantic search over patient notes ---
{vector_evidence}

Question: {question}

Answer:"""


def get_llm():
    return ChatAnthropic(model=CLAUDE_MODEL, max_tokens=600)


def get_sql_database():
    return SQLDatabase.from_uri(f"sqlite:///{SQLITE_PATH}")


def get_graph():
    uri = os.environ["NEO4J_URI"].replace("neo4j+s://", "neo4j://")
    return Neo4jGraph(
        url=uri,
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        driver_config={"encrypted": True, "trusted_certificates": TrustCustomCAs(certifi.where())},
    )


def get_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(PINECONE_INDEX_NAME)


def get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _strip_query(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    return text.strip()


def query_sql(question, db, llm):
    raw = llm.invoke(SQL_PROMPT.format(schema=db.get_table_info(), question=question)).content
    query = _strip_query(raw)
    if query.upper() == "NONE":
        return None, None
    try:
        result = db.run(query)
        return query, result
    except Exception as e:
        return query, f"(query failed: {e})"


def query_graph(question, graph, llm):
    raw = llm.invoke(CYPHER_PROMPT.format(schema=graph.schema, question=question)).content
    query = _strip_query(raw)
    if query.upper() == "NONE":
        return None, None
    try:
        result = graph.query(query)
        return query, result
    except Exception as e:
        return query, f"(query failed: {e})"


def query_vectors(question, index, embedder, top_k=3):
    vector = embedder.embed_query(question)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results["matches"]


def answer_question(question, db, graph, index, embedder, llm):
    sql_query, sql_result = query_sql(question, db, llm)
    cypher_query, cypher_result = query_graph(question, graph, llm)
    matches = query_vectors(question, index, embedder)

    sql_evidence = f"Query: {sql_query}\nResult: {sql_result}" if sql_query else "(not used for this question)"
    graph_evidence = f"Query: {cypher_query}\nResult: {cypher_result}" if cypher_query else "(not used for this question)"
    vector_evidence = "\n\n".join(
        f"[{m['metadata']['patient_name']}] {m['metadata']['text']}" for m in matches
    ) or "(no results)"

    prompt = SYNTHESIS_PROMPT.format(
        sql_evidence=sql_evidence,
        graph_evidence=graph_evidence,
        vector_evidence=vector_evidence,
        question=question,
    )
    answer = llm.invoke(prompt).content

    evidence = {
        "sql_query": sql_query,
        "sql_result": sql_result,
        "cypher_query": cypher_query,
        "cypher_result": cypher_result,
        "vector_matches": matches,
    }
    return answer, evidence


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Which patients on insulin also have other unrelated conditions?"

    db = get_sql_database()
    graph = get_graph()
    index = get_pinecone_index()
    embedder = get_embedder()
    llm = get_llm()

    answer, evidence = answer_question(question, db, graph, index, embedder, llm)

    print(f"Question: {question}\n")
    print(f"--- SQL ---\n{evidence['sql_query']}\n{evidence['sql_result']}\n")
    print(f"--- Cypher ---\n{evidence['cypher_query']}\n{evidence['cypher_result']}\n")
    print(f"--- Vector matches ---")
    for m in evidence["vector_matches"]:
        print(f"  {m['metadata']['patient_name']} (score={m['score']:.4f})")
    print(f"\nAnswer:\n{answer}")
