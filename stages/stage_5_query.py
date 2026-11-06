"""
STAGE 5 — QUERY TRANSFORMATION
==============================
Improvement over Stage 4:  retrieve with the raw query  ->  transform the query first.

THE PROBLEM with Stage 4:
    Retrieval quality is capped by the QUERY itself. A user query is short,
    informal, and often phrased nothing like the source documents:
        user: "why is BM25 good for weird words?"
        doc : "sparse retrieval excels at matching rare terms and identifiers"
    No retriever can fix a query that simply does not overlap the answer.

THE FIX — three query-transformation techniques, applied before retrieval:

    1. QUERY REWRITING / EXPANSION
       The LLM rewrites the query into a fuller, more formal form and adds
       synonyms, so it overlaps document vocabulary better.

    2. MULTI-QUERY
       The query is decomposed into sub-questions. Each sub-question is
       retrieved independently; results are fused. Broadens recall and
       handles multi-part questions that one embedding cannot cover.

    3. HyDE (Hypothetical Document Embeddings)
       The LLM drafts a hypothetical ANSWER to the question. We embed THAT
       (not the question) and retrieve with it. A hypothetical answer is
       written in document-style prose, so its embedding lands near the real
       answer chunks.

All variant queries feed the Stage-4 retrieve+rerank engine; their results
are fused with RRF.

Run it:
    python stages/stage_5_query.py
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
RERANK_POOL = 16
MIN_RATIO = 0.45


class RAGPipeline:
    def __init__(self):
        self.embedder = get_embedder()
        self.llm = get_llm()
        self.dense = VectorStore(self.embedder)
        self.sparse = BM25()
        self._ingest()
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

    # ---- query transformation -------------------------------------------
    def _transform(self, question):
        """Produce a set of query variants from the original question."""
        variants = [question]

        # 1. rewrite / expand
        rewritten = self.llm.complete(f"Rewrite this search query.\nQuery: {question}",
                                      task="rewrite")
        if rewritten and rewritten not in variants:
            variants.append(rewritten)

        # 2. multi-query decomposition
        subs = self.llm.complete(f"Decompose into sub-questions.\nQuestion: {question}",
                                 task="decompose")
        for s in subs:
            if s and s not in variants:
                variants.append(s)

        return variants

    def _hyde_vector(self, question):
        """Draft a hypothetical answer and embed it for dense retrieval."""
        pseudo_doc = self.llm.complete(
            f"Write a hypothetical passage answering this.\nQuestion: {question}",
            task="hyde")
        return self.embedder.embed(pseudo_doc)

    # ---- retrieval -------------------------------------------------------
    def _retrieve_for(self, query, hyde_vec=None):
        dense_hits = self.dense.search(query, k=CANDIDATES, query_vec=hyde_vec)
        sparse_hits = self.sparse.search(query, k=CANDIDATES)
        return reciprocal_rank_fusion([dense_hits, sparse_hits], top_n=CANDIDATES)

    def query(self, question):
        # 1. build query variants + a HyDE embedding
        variants = self._transform(question)
        hyde_vec = self._hyde_vector(question)

        # 2. retrieve for every variant, plus one HyDE-vector retrieval
        result_lists = [self._retrieve_for(v) for v in variants]
        result_lists.append(self._retrieve_for(question, hyde_vec=hyde_vec))

        # 3. fuse all variant results, then rerank against the ORIGINAL query
        fused = reciprocal_rank_fusion(result_lists, top_n=RERANK_POOL)
        reranked = self.reranker.rerank(question, fused, top_k=TOP_K,
                                        min_ratio=MIN_RATIO)
        contexts = [h["text"] for h in reranked]

        # 4. generate
        prompt = build_prompt(question, contexts)
        answer = self.llm.complete(prompt, task="answer",
                                   question=question, context="\n\n".join(contexts))
        return {"answer": answer, "contexts": contexts, "variants": variants}


def _demo():
    pipe = RAGPipeline()
    print(f"[Stage 5: Query transformation]  chunks: {pipe.n_chunks}  LLM: {pipe.llm.name}\n")
    for q in [
        "What is HyDE and why does it improve retrieval?",
        "Why does sparse retrieval like BM25 beat embeddings on rare terms?",
    ]:
        out = pipe.query(q)
        print(f"Q: {q}")
        print(f"   variants used: {out['variants']}")
        print(f"A: {out['answer']}\n")


if __name__ == "__main__":
    _demo()
