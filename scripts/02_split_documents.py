import importlib.util
import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _import_by_path(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SCRIPTS_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_documents = _import_by_path("doc_loader", "01_load_documents.py").load_documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    documents = load_documents()
    chunks = split_documents(documents)

    print(f"{len(documents)} pages -> {len(chunks)} chunks\n")

    sample = chunks[5]
    print("--- sample chunk ---")
    print("metadata:", sample.metadata)
    print("length (chars):", len(sample.page_content))
    print("content:")
    print(sample.page_content)
