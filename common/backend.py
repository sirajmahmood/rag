"""
common/backend.py
-----------------
Pluggable LLM + embedding backend.

By default this runs FULLY OFFLINE using deterministic mock implementations,
so the whole project is runnable with `python stages/stage_X.py` and no setup.

If you set ANTHROPIC_API_KEY in your environment, the LLM calls automatically
use the real Anthropic API. Embeddings always use a local hashing embedder by
default (no embedding API needed); set USE_ST=1 and `pip install
sentence-transformers` to use real neural embeddings instead.

The point of the project is to *see the RAG architecture improve*, which the
mock backend demonstrates clearly because retrieval quality is real (it is just
the text generation that is mocked).
"""

import os
import re
import math
import hashlib
from collections import Counter

# --------------------------------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------------------------------

_DIM = 256


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class HashingEmbedder:
    """
    A deterministic bag-of-words hashing embedder.

    It is NOT as good as a neural embedder, but it produces real semantic
    overlap: documents sharing vocabulary land near each other. That is enough
    to make retrieval improvements visible. Vectors are L2-normalized so dot
    product equals cosine similarity.
    """

    dim = _DIM

    def embed(self, text):
        vec = [0.0] * _DIM
        toks = _tokenize(text)
        if not toks:
            return vec
        tf = Counter(toks)
        for tok, count in tf.items():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % _DIM
            sign = 1.0 if (h >> 8) & 1 else -1.0
            # sublinear term frequency damping
            vec[idx] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


class STEmbedder:
    """Real neural embeddings via sentence-transformers (optional)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_many(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder():
    if os.environ.get("USE_ST") == "1":
        try:
            return STEmbedder()
        except Exception as e:  # pragma: no cover
            print(f"[backend] sentence-transformers unavailable ({e}); using hashing embedder")
    return HashingEmbedder()


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

class MockLLM:
    """
    Offline stand-in for a chat model.

    It is deliberately simple but *context-aware*: for answer generation it
    extracts the sentences from the supplied context that best overlap the
    question. This means a better-retrieved context produces a better mock
    answer, so improvements in the retrieval stack still show up end-to-end.

    It also supports the auxiliary tasks the later stages need: query
    rewriting, sub-question decomposition, HyDE drafting, relevance grading,
    and sufficiency judging -- all via lightweight heuristics.
    """

    name = "MockLLM(offline)"

    # -- generic completion --------------------------------------------------
    def complete(self, prompt, task=None, **kw):
        task = task or _infer_task(prompt)
        handler = {
            "answer": self._answer,
            "rewrite": self._rewrite,
            "decompose": self._decompose,
            "hyde": self._hyde,
            "context_summary": self._context_summary,
            "grade": self._grade,
            "sufficient": self._sufficient,
        }.get(task, self._answer)
        return handler(prompt, **kw)

    # -- task handlers -------------------------------------------------------
    def _answer(self, prompt, question=None, context=None, **kw):
        question = question or _extract(prompt, "Question:")
        context = context or _extract(prompt, "Context:")
        if not context.strip():
            return "I don't have enough information in the provided context to answer that."
        q_toks = set(_tokenize(question))
        sentences = _split_sentences(context)
        scored = []
        for s in sentences:
            s_toks = set(_tokenize(s))
            if not s_toks:
                continue
            overlap = len(q_toks & s_toks) / (len(q_toks) + 1e-9)
            scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])
        top = [s for ov, s in scored[:3] if ov > 0]
        if not top:
            return "The retrieved context does not appear to address the question."
        return " ".join(top)

    def _rewrite(self, prompt, **kw):
        q = _extract(prompt, "Query:") or prompt
        q = q.strip().rstrip("?")
        # heuristic expansion: formalize + add a synonym hint
        expansions = {
            "rag": "retrieval augmented generation",
            "llm": "large language model",
            "embedding": "vector embedding semantic representation",
            "rerank": "reranking cross-encoder relevance scoring",
            "chunk": "chunk chunking document splitting",
            "hybrid": "hybrid dense sparse retrieval",
        }
        extra = []
        for k, v in expansions.items():
            if k in q.lower():
                extra.append(v)
        return q + ((" " + " ".join(extra)) if extra else "")

    def _decompose(self, prompt, **kw):
        q = _extract(prompt, "Question:") or prompt
        # split on conjunctions / question marks into sub-questions
        parts = re.split(r"\band\b|\bvs\b|\bversus\b|\bcompared to\b|\?", q.lower())
        subs = [p.strip() for p in parts if len(p.strip()) > 8]
        if len(subs) <= 1:
            return [q.strip()]
        return [s if s.endswith("?") else s + "?" for s in subs]

    def _hyde(self, prompt, **kw):
        q = _extract(prompt, "Question:") or prompt
        # produce a pseudo-document: declarative restatement padded with the
        # query's own key terms so its embedding sits near real documents
        toks = _tokenize(q)
        body = " ".join(toks)
        return (
            f"{q.strip().rstrip('?')}. This document explains {body}. "
            f"It describes how {body} works and why {body} matters in practice."
        )

    def _context_summary(self, prompt, **kw):
        doc_title = _extract(prompt, "Document:") or "the source document"
        chunk = _extract(prompt, "Chunk:") or ""
        first = _split_sentences(chunk)[:1]
        topic = first[0][:80] if first else "this section"
        return f"This chunk is from {doc_title.strip()} and covers: {topic}"

    def _grade(self, prompt, question=None, context=None, **kw):
        question = question or _extract(prompt, "Question:")
        context = context or _extract(prompt, "Context:")
        q_toks = set(_tokenize(question))
        c_toks = set(_tokenize(context))
        if not q_toks:
            return "yes"
        overlap = len(q_toks & c_toks) / len(q_toks)
        return "yes" if overlap >= 0.34 else "no"

    def _sufficient(self, prompt, question=None, context=None, **kw):
        # judges whether context covers enough of the question's key terms
        question = question or _extract(prompt, "Question:")
        context = context or _extract(prompt, "Context:")
        q_toks = [t for t in _tokenize(question) if len(t) > 3]
        c_toks = set(_tokenize(context))
        if not q_toks:
            return "yes"
        covered = sum(1 for t in q_toks if t in c_toks) / len(q_toks)
        return "yes" if covered >= 0.6 else "no"


class AnthropicLLM:  # pragma: no cover - requires network + key
    """Real Anthropic-backed LLM. Used automatically if ANTHROPIC_API_KEY set."""

    name = "AnthropicLLM(claude)"

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = os.environ.get("RAG_MODEL", "claude-haiku-4-5-20251001")

    def complete(self, prompt, task=None, **kw):
        system = {
            "rewrite": "Rewrite the search query to maximize retrieval recall. Output only the rewritten query.",
            "decompose": "Decompose the question into 1-4 atomic sub-questions, one per line. No numbering.",
            "hyde": "Write a short hypothetical passage that would answer the question. Output only the passage.",
            "context_summary": "Write one sentence situating this chunk within its document. Output only the sentence.",
            "grade": "Is the context relevant to the question? Answer only 'yes' or 'no'.",
            "sufficient": "Does the context contain enough information to fully answer the question? Answer only 'yes' or 'no'.",
        }.get(task, "Answer the question using only the provided context. If the context is insufficient, say so.")
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        if task == "decompose":
            return [l.strip(" -*0123456789.") for l in text.splitlines() if l.strip()]
        return text


def get_llm():
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception as e:  # pragma: no cover
            print(f"[backend] Anthropic unavailable ({e}); using MockLLM")
    return MockLLM()


# --------------------------------------------------------------------------
# small text helpers
# --------------------------------------------------------------------------

def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract(prompt, marker):
    """Pull the text after a marker like 'Question:' up to the next marker."""
    idx = prompt.find(marker)
    if idx == -1:
        return ""
    rest = prompt[idx + len(marker):]
    nxt = re.search(r"\n[A-Z][a-zA-Z ]+:", rest)
    return rest[: nxt.start()].strip() if nxt else rest.strip()


def _infer_task(prompt):
    p = prompt.lower()
    if "rewrite" in p and "query" in p:
        return "rewrite"
    if "sub-question" in p or "decompose" in p:
        return "decompose"
    if "hypothetical" in p:
        return "hyde"
    if "situate" in p or "chunk:" in p and "document:" in p:
        return "context_summary"
    if "relevant" in p and ("yes" in p or "no" in p):
        return "grade"
    if "sufficient" in p or "enough information" in p:
        return "sufficient"
    return "answer"
