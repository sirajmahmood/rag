"""
STAGE 6 — CONTEXTUAL INGESTION + SMART ASSEMBLY
===============================================
Improvement over Stage 5:  improve the INDEX and the PROMPT, not just the query.

Stages 2-5 all improved RETRIEVAL. Stage 6 improves the two things on either
side of it: what goes INTO the index, and how chunks are ARRANGED in the
prompt. Four techniques:

    1. CONTEXTUAL INGESTION  (Anthropic's method)
       A chunk like "It excels at matching paraphrases" is useless out of
       context -- what is "it"? Before indexing, the LLM writes one sentence
       situating the chunk in its document, and we PREPEND it. The embedded
       and BM25-indexed text is now self-contained, fixing a major class of
       retrieval misses. Done once at ingest time => zero query-time cost.

    2. PARENT-CHILD CHUNKING
       We EMBED small, precise child chunks (sharp retrieval) but RETURN the
       larger parent chunk to the LLM (rich context). Best of both chunk
       sizes at once.

    3. MMR DIVERSITY
       After reranking, near-duplicate chunks waste context space. Maximal
       Marginal Relevance picks chunks that are relevant AND diverse, so the
       4 chunks cover more of the answer.

    4. LOST-IN-THE-MIDDLE ORDERING
       LLMs attend most to the start and end of context. We place the
       strongest chunks at the edges, weaker ones in the middle.

Run it:
    python stages/stage_6_contextual.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.backend import get_embedder, get_llm
from common.reranker import get_reranker
from common.utils import (load_documents, chunk_parent_child, VectorStore, BM25,
                          reciprocal_rank_fusion, mmr, build_prompt)

TOP_K = 4
CANDIDATES = 12
RERANK_POOL = 16
MMR_POOL = 8     # rerank down to this, then MMR-select TOP_K from it
MIN_RATIO = 0.45 # drop chunks scoring below 45% of the top chunk's score


class RAGPipeline:
    def __init__(self):
        self.embedder = get_embedder()
        self.llm = get_llm()
        self.dense = VectorStore(self.embedder)
        self.sparse = BM25()
        self._ingest()
        # fit the reranker on the child chunks (the unit it scores)
        self.reranker = get_reranker(self._child_texts)

    # ---- contextual ingestion -------------------------------------------
    def _contextualize(self, doc_title, child_chunk):
        """LLM writes one sentence situating the chunk in its document."""
        prompt = (f"Situate this chunk within its document.\n"
                  f"Document: {doc_title}\nChunk: {child_chunk}")
        return self.llm.complete(prompt, task="context_summary")

    def _ingest(self):
        docs = load_documents()
        embed_texts, return_texts, metas = [], [], []
        for doc in docs:
            # parent-child chunking
            for ci, pair in enumerate(chunk_parent_child(doc["text"],
                                                         parent_max_words=220,
                                                         child_max_words=70)):
                child, parent = pair["child"], pair["parent"]
                # contextual ingestion: prepend a situating sentence to the
                # text we INDEX (embed + BM25), improving retrievability
                context_line = self._contextualize(doc["title"], child)
                indexed = f"{context_line}\n{child}"

                embed_texts.append(indexed)      # what we search against
                return_texts.append(parent)      # what the LLM receives
                metas.append({"doc": doc["id"], "chunk_index": ci, "child": child})

        # metas carry parent text + the precise child that was matched, so
        # every hit can map back to its parent regardless of which retriever
        # surfaced it.
        full_metas = [{**m, "parent": rt} for m, rt in zip(metas, return_texts)]
        # dense store: embed the contextualized child, but RETURN the parent
        self.dense.add(return_texts, metas=full_metas, embed_texts=embed_texts)
        # sparse store: BM25 scores the contextualized text directly
        self.sparse.add(embed_texts, metas=full_metas)
        self._child_texts = [m["child"] for m in metas]
        self.n_chunks = len(embed_texts)

    def query(self, question):
        # 1. hybrid retrieve. Normalize BOTH retrievers to return the parent
        #    text, so reranking/MMR/fusion all operate on the same unit.
        dense_raw = self.dense.search(question, k=CANDIDATES)
        sparse_raw = self.sparse.search(question, k=CANDIDATES)
        dense_hits = [{"text": h["meta"]["parent"], "score": h["score"],
                       "meta": h["meta"]} for h in dense_raw]
        sparse_hits = [{"text": h["meta"]["parent"], "score": h["score"],
                        "meta": h["meta"]} for h in sparse_raw]

        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], top_n=RERANK_POOL)

        # 2. rerank the pool, applying a score threshold so off-topic chunks
        #    are dropped before MMR. MMR then diversifies among the chunks
        #    that survived -- it never has to pad the result with noise.
        reranked = self.reranker.rerank(question, fused, top_k=MMR_POOL,
                                        score_field="child", min_ratio=MIN_RATIO)
        # 3. MMR-select for diversity from the surviving relevant chunks
        q_vec = self.embedder.embed(question)
        k = min(TOP_K, len(reranked))
        diverse = mmr(q_vec, reranked, self.embedder, k=k, lambda_=0.85)
        contexts = [h["text"] for h in diverse]
        # the precise child chunks that were actually matched (for evaluation)
        units = [h["meta"].get("child", h["text"]) for h in diverse]

        # 4. assemble prompt with lost-in-the-middle ordering
        prompt = build_prompt(question, contexts, lost_in_middle=True)
        answer = self.llm.complete(prompt, task="answer",
                                   question=question, context="\n\n".join(contexts))
        return {"answer": answer, "contexts": contexts, "retrieval_units": units}


def _demo():
    pipe = RAGPipeline()
    print(f"[Stage 6: Contextual ingest + MMR]  chunks: {pipe.n_chunks}  LLM: {pipe.llm.name}\n")
    for q in [
        "What does contextual ingestion do to a chunk before storing it?",
        "Why are chunks placed at the start and end of the context window?",
    ]:
        out = pipe.query(q)
        print(f"Q: {q}")
        print(f"A: {out['answer']}\n")


if __name__ == "__main__":
    _demo()
