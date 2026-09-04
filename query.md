# Test Queries — Attention Is All You Need RAG Demo

Use these in the Streamlit app (`app.py`) or via `python scripts\07_rag_chain.py "<question>"`.

1. What is scaled dot-product attention and why is it scaled by the square root of dk?
2. How many attention heads did the authors use, and how many encoder layers are in the model?
3. What BLEU score did the Transformer achieve on English-to-German translation?
4. How does the computational complexity of self-attention compare to recurrent layers?
5. Why does the Transformer need positional encoding if it has no recurrence or convolution?
6. What dataset was used to train the model, and how large was it?
7. How well does the Transformer perform on English constituency parsing compared to other models?
8. What programming language did the authors use to implement the Transformer?

Question 8 is deliberately out-of-scope — it's not answered anywhere in the paper. The RAG prompt
tells Claude to answer only from retrieved context and say so if the answer isn't there. If Claude
answers confidently anyway (e.g. guessing "Python"), the "only use context" instruction isn't being
followed strictly enough.
