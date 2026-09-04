import importlib.util
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from pinecone import Pinecone

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_NAME = "attention-paper-rag"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say so plainly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def _import_by_path(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SCRIPTS_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_embedding_model = _import_by_path("embedder", "03_generate_embeddings.py").get_embedding_model


def get_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(INDEX_NAME)


def get_llm():
    return ChatAnthropic(model=CLAUDE_MODEL, max_tokens=500)


def retrieve_chunks(question, embedder, index, top_k=3):
    question_vector = embedder.embed_query(question)
    results = index.query(vector=question_vector, top_k=top_k, include_metadata=True)
    return results["matches"]


def answer_question(question, embedder, index, llm, top_k=3):
    matches = retrieve_chunks(question, embedder, index, top_k=top_k)

    context = "\n\n".join(
        f"[Page {m['metadata']['page_label']}] {m['metadata']['text']}"
        for m in matches
    )
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    response = llm.invoke(prompt)
    return response.content, matches


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    embedder = get_embedding_model()
    index = get_pinecone_index()
    llm = get_llm()

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is multi-head attention?"

    answer, matches = answer_question(question, embedder, index, llm)

    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}\n")
    print("--- sources ---")
    for m in matches:
        print(f"page {m['metadata']['page_label']} (score={m['score']:.4f})")
