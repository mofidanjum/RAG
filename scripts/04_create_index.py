import os
import sys

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "attention-paper-rag"
EMBEDDING_DIMENSION = 384


def get_pinecone_client():
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def ensure_index(pc):
    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"Index '{INDEX_NAME}' already exists, skipping creation.")
        return

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created index '{INDEX_NAME}'.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    pc = get_pinecone_client()
    ensure_index(pc)

    desc = pc.describe_index(INDEX_NAME)
    print("\n--- index description ---")
    print(desc)
