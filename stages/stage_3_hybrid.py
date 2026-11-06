"""
STAGE 3 — HYBRID RETRIEVAL
==========================
Improvement over Stage 2:  dense-only retrieval  ->  hybrid (dense + sparse) with RRF.

THE PROBLEM with Stage 2:
    Dense embedding search matches MEANING. It is great at paraphrases but it
    can miss exact tokens -- acronyms ("BM25", "HyDE", "RRF"), proper nouns,
    identifiers -- because those terms get smeared across the embedding.

THE FIX:
    Run TWO retrievers in parallel:
        - dense  (VectorStore / cosine)  -> semantic matches
        - sparse (BM25)                  -> exact keyword matches
    Then merge their ranked lists with Reciprocal Rank Fusion. RRF needs no
    score normalization: it only uses each item's RANK in each list, so the
    wildly different score scales of cosine vs BM25 do not matter.

This is usually the single biggest, cheapest retrieval-quality win.

Run it:
    python stages/stage_3_hybrid.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.backend import get_embedder, get_llm
from common.utils import (load_documents, chunk_sentences, VectorStore, BM25,
                          reciprocal_rank_fusion, build_prompt)

TOP_K = 4
CANDIDATES = 10  # retrieve a wider net from each retriever before fusion


class RAGPipeline:
    def __init__(self):
        self.embedder = get_embedder()
        self.llm = get_llm()
        self.dense = VectorStore(self.embedder)
        self.sparse = BM25()                       # <-- new: sparse index
        self._ingest()

    def _ingest(self):
        docs = load_documents()
        all_chunks, metas = [], []
        for doc in docs:
            for ci, chunk in enumerate(chunk_sentences(doc["text"],
                                                       max_words=110,
                                                       overlap_sentences=1)):
                all_chunks.append(chunk)
                metas.append({"doc": doc["id"], "chunk_index": ci})
        self.dense.add(all_chunks, metas=metas)
        self.sparse.add(all_chunks, metas=metas)   # <-- index the same chunks for BM25
        self.n_chunks = len(all_chunks)

    def query(self, question):
        # 1. retrieve from BOTH retrievers
        dense_hits = self.dense.search(question, k=CANDIDATES)
        sparse_hits = self.sparse.search(question, k=CANDIDATES)
        # 2. fuse the two ranked lists -- rank-based, no normalization needed
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], top_n=TOP_K)
        contexts = [h["text"] for h in fused]
        # 3. generate
        prompt = build_prompt(question, contexts)
        answer = self.llm.complete(prompt, task="answer",
                                   question=question, context="\n\n".join(contexts))
        return {"answer": answer, "contexts": contexts}


def _demo():
    pipe = RAGPipeline()
    print(f"[Stage 3: Hybrid retrieval]  chunks indexed: {pipe.n_chunks}  LLM: {pipe.llm.name}\n")
    for q in [
        "Why does sparse retrieval like BM25 beat embeddings on rare terms?",
        "How does Reciprocal Rank Fusion merge results without score normalization?",
    ]:
        out = pipe.query(q)
        print(f"Q: {q}")
        print(f"A: {out['answer']}\n")


if __name__ == "__main__":
    _demo()
