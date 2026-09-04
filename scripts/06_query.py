import importlib.util
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_NAME = "attention-paper-rag"

DEFAULT_QUESTION = "Why did the authors choose attention mechanisms over recurrent neural networks?"


def _import_by_path(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SCRIPTS_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_embedding_model = _import_by_path("embedder", "03_generate_embeddings.py").get_embedding_model


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUESTION

    embedder = get_embedding_model()
    question_vector = embedder.embed_query(question)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    results = index.query(vector=question_vector, top_k=3, include_metadata=True)

    print(f"Question: {question}\n")
    for i, match in enumerate(results["matches"]):
        print(f"--- result {i + 1} (score={match['score']:.4f}, page={match['metadata']['page_label']}) ---")
        print(match["metadata"]["text"])
        print()
