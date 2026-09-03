# RAG & GraphRAG — Learning Notes

Building up from first principles, grounded in the `data-agent-poc` project
(`ingestion/load_unstructured.py`, `agent/tools.py`).

---

## Concept 1: Why text needs to become numbers

A computer can't "understand" the meaning of a sentence like *"My album
keeps skipping"*. It can only do math on numbers. So before any search or
comparison can happen, text has to be turned into numbers that somehow
*preserve meaning* — sentences with similar meaning should end up as
similar numbers, even if they don't share a single word.

That's the problem embeddings solve.

## Concept 2: What an embedding actually is

An embedding is **a list of numbers (a vector) that represents a piece of
text's meaning** — in our project, 384 numbers per ticket (the output size
of the `all-MiniLM-L6-v2` model):

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts).tolist()
```

Each ticket's text goes in, a list of 384 floats comes out — e.g.
`[0.12, -0.87, 0.03, ...]`. That's it.

**The key property that makes it useful:** two sentences with similar
*meaning* produce vectors that land close together in that 384-number
space, even with completely different words.

- "My album keeps skipping" and "tracks cut out during playback" → close
  together (similar meaning).
- "My album keeps skipping" and "please update my billing address" → far
  apart (unrelated meaning).

**Where the numbers come from:** the model (`all-MiniLM-L6-v2`) was already
trained by someone else, on huge amounts of text, to learn this
"similar meaning → similar numbers" mapping. We don't train it — we just
use it as-is (a *pre-trained* model).

---

## Concept 3: How do you find the closest vector out of thousands?

Once every ticket is a 384-number vector, a search query gets embedded the
same way, into its own 384-number vector. Now you need to find which
stored vectors are *closest* to the query vector. That's a **vector
database** — its only job.

A vector DB (Chroma, in our project) does two things:

1. **Stores** each vector alongside its original text and metadata.
2. **Searches** — given a query vector, it computes the distance between it
   and every stored vector, and returns the closest ones
   ("nearest neighbors").

From `agent/tools.py`:

```python
query_embedding = model.encode([query]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=n_results)
```

"Distance" here usually means **cosine similarity** — do two vectors point
in a similar direction in that 384-dimensional space? Closer direction =
more similar meaning. You don't need the underlying math to use it — just:
*smaller distance = more similar meaning*.

At small scale (our 33 tickets) this could even be done by comparing every
vector to every other one, brute-force. Vector DBs matter once you have
millions of vectors — they use indexing tricks to avoid checking every
single one.

---

*(Next up: putting Concepts 1–3 together as "RAG" — Retrieval-Augmented
Generation — and where the LLM actually enters the picture.)*
