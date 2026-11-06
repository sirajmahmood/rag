"""
STAGE 2 — BETTER CHUNKING
=========================
Improvement over Stage 1:  fixed-size chunking  ->  sentence-aware chunking.

THE PROBLEM with Stage 1:
    chunk_fixed() splits on a raw word count. A chunk can begin or end in the
    middle of a sentence: "...the model produces an answer grounded in that"
    The embedding of a half-sentence is noisy, and a retrieved half-sentence
    is poor context for the LLM.

THE FIX:
    chunk_sentences() respects sentence boundaries and keeps a one-sentence
    overlap between consecutive chunks, so no idea is split and boundary
    context is preserved. Each chunk is now a clean set of whole sentences.

Everything else is identical to Stage 1, so any metric change is attributable
purely to the chunking strategy.

Run it:
    python stages/stage_2_chunking.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.backend import get_embedder, get_llm
from common.utils import load_documents, chunk_sentences, VectorStore, build_prompt

TOP_K = 4


class RAGPipeline:
    def __init__(self):
        self.embedder = get_embedder()
        self.llm = get_llm()
        self.store = VectorStore(self.embedder)
        self._ingest()

    def _ingest(self):
        docs = load_documents()
        all_chunks, metas = [], []
        for doc in docs:
            # <-- the only change vs Stage 1: sentence-aware chunking
            for ci, chunk in enumerate(chunk_sentences(doc["text"],
                                                       max_words=110,
                                                       overlap_sentences=1)):
                all_chunks.append(chunk)
                metas.append({"doc": doc["id"], "chunk_index": ci})
        self.store.add(all_chunks, metas=metas)
        self.n_chunks = len(all_chunks)

    def query(self, question):
        hits = self.store.search(question, k=TOP_K)
        contexts = [h["text"] for h in hits]
        prompt = build_prompt(question, contexts)
        answer = self.llm.complete(prompt, task="answer",
                                   question=question, context="\n\n".join(contexts))
        return {"answer": answer, "contexts": contexts}


def _demo():
    pipe = RAGPipeline()
    print(f"[Stage 2: Sentence chunking]  chunks indexed: {pipe.n_chunks}  LLM: {pipe.llm.name}\n")
    for q in [
        "What are the three steps of a naive RAG pipeline?",
        "What is the tradeoff between small and large chunks?",
    ]:
        out = pipe.query(q)
        print(f"Q: {q}")
        print(f"A: {out['answer']}\n")


if __name__ == "__main__":
    _demo()
