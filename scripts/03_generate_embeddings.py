import importlib.util
import os
import sys

from langchain_huggingface import HuggingFaceEmbeddings

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _import_by_path(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SCRIPTS_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_documents = _import_by_path("doc_loader", "01_load_documents.py").load_documents
split_documents = _import_by_path("doc_splitter", "02_split_documents.py").split_documents


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL_NAME}'...\n")

    embedder = get_embedding_model()

    sample_texts = [chunks[5].page_content, chunks[6].page_content]
    vectors = embedder.embed_documents(sample_texts)

    print(f"Generated {len(vectors)} vectors")
    print(f"Vector dimension: {len(vectors[0])}")
    print(f"First 5 values of vector 0: {vectors[0][:5]}")
