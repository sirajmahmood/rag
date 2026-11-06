"""
STAGE 4 — RERANKING
===================
Improvement over Stage 3:  trust fused ranking  ->  retrieve wide, then RERANK.

THE PROBLEM with Stage 3:
    Both dense and sparse retrieval are FAST but APPROXIMATE. A bi-encoder
    embeds the query and the document SEPARATELY, so it never actually
    compares them token-by-token. RRF inherits that approximation. The chunk
    that is genuinely most relevant is often in the top-10 but not the top-4.

THE FIX:  two-stage retrieval.
    Stage A (recall):    retrieve a WIDE candidate set (e.g. 12 chunks) with
                         cheap hybrid search.
    Stage B (precision): RERANK those candidates with a stronger model that
                         scores the (query, chunk) pair JOINTLY, then keep
                         only the true top-4.

    A production system uses a cross-encoder (Cohere Rerank, BGE-reranker).

THE RERANKER  (see common/reranker.py)
    `get_reranker()` returns a REAL cross-encoder
    (cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers) when its
    weights are available. A cross-encoder feeds query+chunk through one
    transformer together, modelling their interaction directly.

    When weights cannot be loaded (e.g. an offline/airgapped environment),
    it falls back to an IDF-weighted heuristic. That fallback is deliberately
    good: IDF weighting + length normalization stop a keyword-stuffed document
    header from out-scoring the real answer chunk -- the exact failure mode of
    a naive overlap reranker.

    Either way the interface is identical, and reranking now produces a real
    gain over Stage 3.

Run it:
    python stages/stage_4_rerank.py

Force a specific reranker:
    RAG_RERANKER=cross      python stages/stage_4_rerank.py   # require the model
    RAG_RERANKER=heuristic  python stages/stage_4_rerank.py   # skip model load
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.backend import get_embedder, get_llm
from common.reranker import get_reranker
from common.utils import (load_documents, chunk_sentences, VectorStore, BM25,
                          reciprocal_rank_fusion, build_prompt)

TOP_K = 4
CANDIDATES = 12
RERANK_POOL = 12   # how many fused candidates the reranker sees
MIN_RATIO = 0.45   # drop chunks scoring below 45% of the top chunk's score


class RAGPipeline:
    def __init__(self):
        self.embedder = get_embedder()
        self.llm = get_llm()
        self.dense = VectorStore(self.embedder)
        self.sparse = BM25()
        self._ingest()
        # build the reranker AFTER ingest so the heuristic fallback can fit
        # its IDF statistics on the full chunk collection
        self.reranker = get_reranker(self._all_chunks)

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
        self.sparse.add(all_chunks, metas=metas)
        self._all_chunks = all_chunks
        self.n_chunks = len(all_chunks)

    def query(self, question):
        # 1. WIDE recall pass: hybrid retrieve a large candidate pool
        dense_hits = self.dense.search(question, k=CANDIDATES)
        sparse_hits = self.sparse.search(question, k=CANDIDATES)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], top_n=RERANK_POOL)
        # 2. PRECISION pass: rerank the pool. With score thresholding the
        #    reranker keeps only chunks close to the best one, so a query with
        #    few relevant chunks gets a small, clean context instead of noise.
        reranked = self.reranker.rerank(question, fused, top_k=TOP_K,
                                        min_ratio=MIN_RATIO)
        contexts = [h["text"] for h in reranked]
        # 3. generate
        prompt = build_prompt(question, contexts)
        answer = self.llm.complete(prompt, task="answer",
                                   question=question, context="\n\n".join(contexts))
        return {"answer": answer, "contexts": contexts}


def _demo():
    pipe = RAGPipeline()
    print(f"[Stage 4: Reranking]  chunks indexed: {pipe.n_chunks}  "
          f"LLM: {pipe.llm.name}  reranker: {pipe.reranker.kind}\n")
    for q in [
        "What problem does a cross-encoder reranker solve?",
        "What does corrective RAG do when retrieved chunks are not relevant?",
    ]:
        out = pipe.query(q)
        print(f"Q: {q}")
        print(f"A: {out['answer']}\n")


if __name__ == "__main__":
    _demo()
