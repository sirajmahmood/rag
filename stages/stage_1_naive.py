"""
STAGE 1 — NAIVE RAG
===================
The simplest possible RAG pipeline. This is the baseline everything else is
measured against.

Pipeline:
    fixed-size chunking  ->  dense embedding  ->  top-k cosine retrieval  ->  stuff into prompt  ->  generate

What is DELIBERATELY missing (and gets fixed in later stages):
    - chunks cut sentences in half  -> Stage 2
    - pure semantic search misses exact keywords -> Stage 3
    - no reranking, so approximate ranking stands -> Stage 4
    - the raw query is used as-is -> Stage 5
    - chunks are decontextualized, no diversity control -> Stage 6
    - single-shot: one retrieve, one answer, no iteration -> Stage 7

Run it:
    python stages/stage_1_naive.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.backend import get_embedder, get_llm
from common.utils import load_documents, chunk_fixed, VectorStore, build_prompt

TOP_K = 4


class RAGPipeline:
    """Naive RAG. Exposes .query(question) -> {answer, contexts}."""

    def __init__(self):
        self.embedder = get_embedder()
        self.llm = get_llm()
        self.store = VectorStore(self.embedder)
        self._ingest()

    def _ingest(self):
        docs = load_documents()
        all_chunks, metas = [], []
        for doc in docs:
            for ci, chunk in enumerate(chunk_fixed(doc["text"], size=120, overlap=20)):
                all_chunks.append(chunk)
                metas.append({"doc": doc["id"], "chunk_index": ci})
        self.store.add(all_chunks, metas=metas)
        self.n_chunks = len(all_chunks)

    def query(self, question):
        # 1. retrieve: embed the raw query, cosine top-k
        hits = self.store.search(question, k=TOP_K)
        contexts = [h["text"] for h in hits]
        # 2. generate: stuff all chunks into one prompt
        prompt = build_prompt(question, contexts)
        answer = self.llm.complete(prompt, task="answer",
                                   question=question, context="\n\n".join(contexts))
        return {"answer": answer, "contexts": contexts}


def _demo():
    pipe = RAGPipeline()
    print(f"[Stage 1: Naive RAG]  chunks indexed: {pipe.n_chunks}  LLM: {pipe.llm.name}\n")
    for q in [
        "What are the three steps of a naive RAG pipeline?",
        "What is HyDE and why does it improve retrieval?",
    ]:
        out = pipe.query(q)
        print(f"Q: {q}")
        print(f"A: {out['answer']}")
        print(f"   (retrieved {len(out['contexts'])} chunks)\n")


if __name__ == "__main__":
    _demo()
