import os
import sys

from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "attention_is_all_you_need.pdf")


def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    documents = load_documents()
    print(f"Loaded {len(documents)} documents (one per PDF page)\n")

    sample = documents[0]
    print("--- sample Document ---")
    print("metadata:", sample.metadata)
    print("page_content (first 300 chars):")
    print(sample.page_content[:300])
