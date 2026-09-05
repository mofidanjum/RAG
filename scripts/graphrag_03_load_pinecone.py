import os
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "patients")
NOTES_DIR = os.path.join(DATA_DIR, "notes")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "patient-notes-rag"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 50


def load_notes():
    patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
    visits = pd.read_csv(os.path.join(DATA_DIR, "visits.csv"))
    patient_name = {row["PATIENT_ID"]: f"{row['FIRST']} {row['LAST']}" for _, row in patients.iterrows()}
    visit_patient = dict(zip(visits["VISIT_ID"], visits["PATIENT_ID"]))

    documents = []
    for filename in os.listdir(NOTES_DIR):
        visit_id = filename.removeprefix("note_").removesuffix(".txt")
        patient_id = visit_patient[visit_id]

        loader = TextLoader(os.path.join(NOTES_DIR, filename), encoding="utf-8")
        doc = loader.load()[0]
        doc.metadata["visit_id"] = visit_id
        doc.metadata["patient_id"] = patient_id
        doc.metadata["patient_name"] = patient_name[patient_id]
        documents.append(doc)

    return documents


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    documents = load_notes()
    print(f"Loaded {len(documents)} notes")

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created index '{INDEX_NAME}'.")
    else:
        print(f"Index '{INDEX_NAME}' already exists, reusing it.")
    index = pc.Index(INDEX_NAME)

    texts = [doc.page_content for doc in documents]
    vectors = embedder.embed_documents(texts)

    records = []
    for doc, vector in zip(documents, vectors):
        records.append({
            "id": doc.metadata["visit_id"],
            "values": vector,
            "metadata": {
                "text": doc.page_content,
                "visit_id": doc.metadata["visit_id"],
                "patient_id": doc.metadata["patient_id"],
                "patient_name": doc.metadata["patient_name"],
            },
        })

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"Upserted records {start}-{start + len(batch) - 1}")

    print("\n--- comparison: plain vector search for the multi-hop question ---")
    question = "patient on insulin for diabetes who also has an unrelated condition like sinusitis"
    question_vector = embedder.embed_query(question)
    results = index.query(vector=question_vector, top_k=5, include_metadata=True)

    for i, match in enumerate(results["matches"]):
        print(f"\nresult {i + 1} (score={match['score']:.4f}, patient={match['metadata']['patient_name']})")
        print(match["metadata"]["text"])
