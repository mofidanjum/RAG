# Test Queries — GraphRAG Hybrid Agent (Stage 2)

Use these with `scripts\graphrag_04_hybrid_query.py "<question>"` or the "Ask a question" box in
`scripts\graphrag_app.py`. Each category exercises a different system on purpose — check the SQL /
Cypher / vector evidence panels to see which one(s) actually contributed to the answer.

## 1. Aggregation questions (SQL should carry these)

1. How many visits did each patient have?
2. Which patient has the most distinct conditions on record?
3. How many patients in this dataset are being treated with insulin?
4. What's the most common visit type across all patients?

SQL can count and group; a graph or vector search can't answer "how many" or "which one is highest"
directly.

## 2. Relationship / multi-hop questions (the graph should carry these)

5. **Which patients on insulin also have other, unrelated conditions?** — the flagship question this
   dataset was built around. Compare the graph's answer against a plain semantic search for the same
   question: the graph returns a complete list; vector search mostly returns more insulin visits and
   misses the unrelated conditions, because they're separate notes with no shared vocabulary.
6. For the patient with the most visits, what other conditions and medications do they have besides
   diabetes-related ones?
7. Find any two patients connected through a chain of shared conditions or medications, regardless of
   how many steps it takes. (Tests whether Claude writes a variable-length Cypher path query — SQL
   genuinely can't express "unknown number of hops" cleanly.)

## 3. Semantic / fuzzy questions (vector search should carry these)

8. Find visit notes that describe a check-up or wellness exam.
9. Which visit notes mention a prescription being given, without naming a specific drug?

These don't have one exact structured answer — they're about matching the *meaning* of a phrase across
free-text notes, which is what Pinecone is for.

## 4. Combined questions (need two or more systems together)

10. Summarize a recent visit for the patient with the most distinct conditions, including what was
    diagnosed or prescribed. (SQL/graph to find the patient and visit, vector search or the visit note
    to get the narrative detail.)
11. Which patients share the same medication, and what different conditions are they each being
    treated for? (Graph traversal to find shared medications, then per-patient condition lookup.)

## 5. Deliberately out-of-scope (tests honesty, not capability)

12. What is each patient's blood type?
13. Did any patient have a surgical procedure? (This dataset's `procedures.csv` has zero rows for
    these 15 patients — a real, verifiable "no data" case, not a hypothetical one.)

Good behavior: the agent should say the data doesn't contain this, not guess a plausible-sounding
answer. If it invents a blood type or a procedure, the "answer only from evidence" instruction in
`SYNTHESIS_PROMPT` (in `scripts/graphrag_04_hybrid_query.py`) isn't being followed strictly enough.

## Known issue: inconsistent Cypher matching on question 5

Running question 5 (2026-09-05) surfaced a real reliability gap: Claude generated an **exact-match**
Cypher query — `Medication {name: 'insulin'}` — against medication nodes whose actual names are the
full drug string (e.g. `"insulin human isophane 70 UNT/ML..."`). No node matches exactly, so the graph
returned `[]` even though the data is there. SQL and vector search picked up the slack and the final
answer was still correct (and honestly reported the graph came back empty rather than guessing), but
this means graph results can silently go missing whenever Claude writes an exact match instead of a
fuzzy one (`CONTAINS`, `=~` regex) for a free-text property. Worth revisiting when Stage 3 adds
smarter query generation — a few-shot example or a schema note ("medication/condition names are full
strings, always match with CONTAINS") would likely fix it.
