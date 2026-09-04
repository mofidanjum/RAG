import importlib.util
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_NAME = "attention-paper-rag"
BATCH_SIZE = 50


def _import_by_path(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SCRIPTS_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_documents = _import_by_path("doc_loader", "01_load_documents.py").load_documents
split_documents = _import_by_path("doc_splitter", "02_split_documents.py").split_documents
get_embedding_model = _import_by_path("embedder", "03_generate_embeddings.py").get_embedding_model


def build_vectors(chunks, embedder):
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedder.embed_documents(texts)

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": f"chunk-{i}",
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "page": chunk.metadata.get("page"),
                "page_label": chunk.metadata.get("page_label"),
            },
        })
    return records


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages -> {len(chunks)} chunks")

    embedder = get_embedding_model()
    records = build_vectors(chunks, embedder)
    print(f"Built {len(records)} vector records\n")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"Upserted records {start}-{start + len(batch) - 1}")

    stats = index.describe_index_stats()
    print("\n--- index stats after upsert ---")
    print(stats)
