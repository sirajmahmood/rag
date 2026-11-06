"""
STAGE 7 — AGENTIC RAG
=====================
Improvement over Stage 6:  one-shot pipeline  ->  an iterative reasoning LOOP.

THE PROBLEM with Stages 1-6:
    Every previous stage is SINGLE-SHOT: retrieve once, answer once. If the
    first retrieval misses, the answer is simply wrong -- nothing checks it
    and nothing tries again. Multi-part questions are especially fragile.

THE FIX — an agent that loops: PLAN -> RETRIEVE -> GRADE -> JUDGE -> (repeat).

    1. PLAN          Decompose the question into sub-questions to chase down.
    2. RETRIEVE      For each open sub-question, run the full Stage-6 engine
                     (hybrid + rerank + MMR).
    3. GRADE         (Corrective-RAG) An LLM grader checks each retrieved
                     chunk for relevance; irrelevant chunks are dropped.
    4. JUDGE         (Self-RAG style) The agent asks "is the gathered context
                     SUFFICIENT to answer?" If yes -> synthesize. If no ->
                     REWRITE the open query and loop again.
    5. STOP          The loop ends on sufficiency or a max-iteration budget,
                     so it can never run forever.

This combines the techniques named ReAct, Corrective-RAG, and Self-RAG into
one transparent loop. Set verbose=True to watch the agent's trace.

Run it:
    python stages/stage_7_agentic.py
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
MMR_POOL = 8
MAX_ITERS = 3          # the agent's step budget -- guarantees termination


class RetrievalEngine:
    """The full Stage-6 retrieve+rerank+MMR engine, reused by the agent."""

    def __init__(self):
        self.embedder = get_embedder()
        self.llm = get_llm()
        self.dense = VectorStore(self.embedder)
        self.sparse = BM25()
        self._ingest()
        self.reranker = get_reranker(self._child_texts)

    def _contextualize(self, doc_title, child):
        return self.llm.complete(
            f"Situate this chunk within its document.\n"
            f"Document: {doc_title}\nChunk: {child}", task="context_summary")

    def _ingest(self):
        docs = load_documents()
        embed_texts, return_texts, metas = [], [], []
        for doc in docs:
            for ci, pair in enumerate(chunk_parent_child(doc["text"],
                                                         parent_max_words=220,
                                                         child_max_words=70)):
                context_line = self._contextualize(doc["title"], pair["child"])
                embed_texts.append(f"{context_line}\n{pair['child']}")
                return_texts.append(pair["parent"])
                metas.append({"doc": doc["id"], "chunk_index": ci,
                              "child": pair["child"]})
        full_metas = [{**m, "parent": rt} for m, rt in zip(metas, return_texts)]
        self.dense.add(return_texts, metas=full_metas, embed_texts=embed_texts)
        self.sparse.add(embed_texts, metas=full_metas)
        self._child_texts = [m["child"] for m in metas]
        self.n_chunks = len(embed_texts)

    def retrieve(self, query):
        dense_raw = self.dense.search(query, k=CANDIDATES)
        sparse_raw = self.sparse.search(query, k=CANDIDATES)
        dense_hits = [{"text": h["meta"]["parent"], "score": h["score"],
                       "meta": h["meta"]} for h in dense_raw]
        sparse_hits = [{"text": h["meta"]["parent"], "score": h["score"],
                        "meta": h["meta"]} for h in sparse_raw]
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], top_n=RERANK_POOL)
        reranked = self.reranker.rerank(query, fused, top_k=MMR_POOL,
                                        score_field="child")
        q_vec = self.embedder.embed(query)
        return mmr(q_vec, reranked, self.embedder, k=TOP_K, lambda_=0.85)


class RAGPipeline:
    """Agentic RAG. Same .query() interface as every other stage."""

    def __init__(self, verbose=False):
        self.engine = RetrievalEngine()
        self.llm = self.engine.llm
        self.verbose = verbose
        self.n_chunks = self.engine.n_chunks

    def _log(self, msg):
        if self.verbose:
            print(f"   [agent] {msg}")

    # ---- the agentic loop ------------------------------------------------
    def query(self, question):
        # STEP 1 -- PLAN: decompose into sub-questions
        sub_questions = self.llm.complete(
            f"Decompose into sub-questions.\nQuestion: {question}", task="decompose")
        self._log(f"plan: {sub_questions}")

        gathered = []          # accumulated, deduplicated parent context
        gathered_units = []    # the matched child chunks (for evaluation)
        seen = set()

        for sub_q in sub_questions:
            open_query = sub_q
            for it in range(1, MAX_ITERS + 1):
                # STEP 2 -- RETRIEVE
                hits = self.engine.retrieve(open_query)

                # STEP 3 -- GRADE each chunk (Corrective RAG)
                kept = []
                for h in hits:
                    verdict = self.llm.complete(
                        f"Is the context relevant? yes or no.\n"
                        f"Question: {sub_q}\nContext: {h['text']}",
                        task="grade", question=sub_q, context=h["text"])
                    if verdict.strip().lower().startswith("y"):
                        kept.append(h)
                self._log(f"sub-q '{sub_q[:40]}' iter {it}: "
                          f"{len(kept)}/{len(hits)} chunks passed grading")

                for h in kept:
                    if h["text"] not in seen:
                        seen.add(h["text"])
                        gathered.append(h["text"])
                        gathered_units.append(h["meta"].get("child", h["text"]))

                # STEP 4 -- JUDGE sufficiency (Self-RAG style)
                ctx = "\n\n".join(gathered)
                verdict = self.llm.complete(
                    f"Is the context sufficient to answer? yes or no.\n"
                    f"Question: {sub_q}\nContext: {ctx}",
                    task="sufficient", question=sub_q, context=ctx)
                if verdict.strip().lower().startswith("y"):
                    self._log(f"sub-q '{sub_q[:40]}': sufficient, moving on")
                    break

                # STEP 5 -- not sufficient: REWRITE the query and loop
                open_query = self.llm.complete(
                    f"Rewrite this search query.\nQuery: {sub_q}", task="rewrite")
                self._log(f"insufficient -> rewritten query: '{open_query[:50]}'")

        # FINAL SYNTHESIS
        # The loop gathered chunks across several sub-questions, so the pool
        # also holds chunks relevant only to a sub-question. Before answering,
        # rerank the whole gathered pool against the ORIGINAL question and
        # apply the score threshold, so the final context is the chunks that
        # actually bear on what the user asked. Best chunks at the edges.
        pool = [{"text": t, "score": 0.0,
                 "meta": {"child": u}}
                for t, u in zip(gathered, gathered_units)]
        if pool:
            final = self.engine.reranker.rerank(question, pool, top_k=6,
                                                score_field="child",
                                                min_ratio=0.45)
        else:
            final = []
        contexts = [h["text"] for h in final]
        units = [h["meta"]["child"] for h in final]
        prompt = build_prompt(question, contexts, lost_in_middle=True)
        answer = self.llm.complete(prompt, task="answer",
                                   question=question, context="\n\n".join(contexts))
        return {"answer": answer, "contexts": contexts, "retrieval_units": units}


def _demo():
    pipe = RAGPipeline(verbose=True)
    print(f"[Stage 7: Agentic RAG]  chunks: {pipe.n_chunks}  LLM: {pipe.llm.name}\n")
    for q in [
        "How does an agentic RAG loop decide whether to retrieve again?",
        "What problem does a cross-encoder reranker solve and what is HyDE?",
    ]:
        print(f"Q: {q}")
        out = pipe.query(q)
        print(f"A: {out['answer']}")
        print(f"   (synthesized from {len(out['contexts'])} gathered chunks)\n")


if __name__ == "__main__":
    _demo()
