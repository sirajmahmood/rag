"""
common/utils.py
---------------
Shared building blocks used across every stage:

  - document loading from the data/ folder
  - chunking strategies (fixed, sentence-aware, parent-child)
  - an in-memory vector store with cosine search
  - a BM25 sparse index
  - Reciprocal Rank Fusion
  - prompt assembly

Each stage imports only the pieces it needs, so you can read a stage top to
bottom and see exactly which techniques it uses.
"""

import os
import re
import math
import glob
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# --------------------------------------------------------------------------
# LOADING
# --------------------------------------------------------------------------

def load_documents(data_dir=DATA_DIR):
    """Load every .txt/.md file in the data folder as {id, title, text}."""
    docs = []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")) +
                   glob.glob(os.path.join(data_dir, "*.md")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({
            "id": os.path.basename(path),
            "title": os.path.basename(path),
            "text": text,
        })
    if not docs:
        raise FileNotFoundError(f"No .txt or .md files found in {data_dir}")
    return docs


# --------------------------------------------------------------------------
# CHUNKING
# --------------------------------------------------------------------------

def _words(text):
    return text.split()


def chunk_fixed(text, size=120, overlap=20):
    """Naive fixed-size word-window chunking with overlap."""
    words = _words(text)
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += step
    return chunks


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(text, max_words=110, overlap_sentences=1):
    """
    Sentence-aware chunking: never splits a sentence in half, and keeps a
    little overlap so context is not lost at chunk boundaries.
    """
    sents = split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        w = len(_words(s))
        if cur and cur_len + w > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_len = sum(len(_words(x)) for x in cur)
        cur.append(s)
        cur_len += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_parent_child(text, parent_max_words=220, child_max_words=70):
    """
    Hierarchical chunking. Returns a list of dicts:
      {child: <small text>, parent: <larger surrounding text>}
    Retrieval matches on the small child; generation receives the parent.
    """
    sents = split_sentences(text)
    # build parents first
    parents, cur, cur_len = [], [], 0
    for s in sents:
        w = len(_words(s))
        if cur and cur_len + w > parent_max_words:
            parents.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += w
    if cur:
        parents.append(" ".join(cur))
    # build children inside each parent
    out = []
    for parent in parents:
        for child in chunk_sentences(parent, max_words=child_max_words, overlap_sentences=0):
            out.append({"child": child, "parent": parent})
    return out


# --------------------------------------------------------------------------
# VECTOR STORE  (dense retrieval)
# --------------------------------------------------------------------------

class VectorStore:
    """In-memory vector store. Records carry arbitrary metadata."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.records = []  # {text, vec, meta}

    def add(self, texts, metas=None, embed_texts=None):
        """
        texts        : the text to RETURN on retrieval
        embed_texts  : the text to EMBED (defaults to texts) -- lets you embed
                       a child chunk but return its parent, etc.
        """
        metas = metas or [{} for _ in texts]
        embed_texts = embed_texts or texts
        vecs = self.embedder.embed_many(embed_texts)
        for t, v, m in zip(texts, vecs, metas):
            self.records.append({"text": t, "vec": v, "meta": m})

    def search(self, query, k=5, query_vec=None):
        qv = query_vec if query_vec is not None else self.embedder.embed(query)
        scored = []
        for r in self.records:
            s = sum(x * y for x, y in zip(qv, r["vec"]))
            scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        return [{"text": r["text"], "score": s, "meta": r["meta"]}
                for s, r in scored[:k]]


# --------------------------------------------------------------------------
# BM25  (sparse retrieval)
# --------------------------------------------------------------------------

def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Classic BM25 sparse retriever."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = []          # list of {text, meta}
        self.tokenized = []
        self.df = Counter()
        self.avgdl = 0.0

    def add(self, texts, metas=None):
        metas = metas or [{} for _ in texts]
        for t, m in zip(texts, metas):
            toks = _tok(t)
            self.docs.append({"text": t, "meta": m})
            self.tokenized.append(toks)
            for term in set(toks):
                self.df[term] += 1
        total = sum(len(t) for t in self.tokenized)
        self.avgdl = total / max(1, len(self.tokenized))

    def search(self, query, k=5):
        q_terms = _tok(query)
        N = len(self.docs)
        scored = []
        for i, toks in enumerate(self.tokenized):
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (N - self.df[term] + 0.5) / (self.df[term] + 0.5))
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * num / den
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return [{"text": self.docs[i]["text"], "score": s, "meta": self.docs[i]["meta"]}
                for s, i in scored[:k]]


# --------------------------------------------------------------------------
# FUSION + RERANK helpers
# --------------------------------------------------------------------------

def reciprocal_rank_fusion(result_lists, k=60, top_n=10):
    """
    Merge multiple ranked result lists into one. Robust, score-free.

    Items are keyed by a stable identifier so that parent-child retrieval
    works correctly: several child chunks can share one parent text, so
    keying on text alone would wrongly collapse distinct chunks. We prefer
    meta['doc'] + meta['chunk_index'] when present, falling back to text.
    """
    def key_of(r):
        m = r.get("meta", {})
        if "doc" in m and "chunk_index" in m:
            return (m["doc"], m["chunk_index"])
        return r["text"]

    scores = defaultdict(float)
    holder = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            key = key_of(r)
            scores[key] += 1.0 / (k + rank + 1)
            holder[key] = r
    merged = sorted(scores.items(), key=lambda x: -x[1])
    out = []
    for key, s in merged[:top_n]:
        rec = dict(holder[key])
        rec["score"] = s
        out.append(rec)
    return out


def mmr(query_vec, candidates, embedder, k=5, lambda_=0.7):
    """
    Maximal Marginal Relevance: pick chunks that are relevant to the query but
    diverse from each other. `candidates` is a list of result dicts.
    """
    pool = []
    for c in candidates:
        v = c["meta"].get("vec") or embedder.embed(c["text"])
        pool.append({**c, "_vec": v})
    selected = []
    while pool and len(selected) < k:
        best, best_score = None, -1e9
        for c in pool:
            relevance = sum(x * y for x, y in zip(query_vec, c["_vec"]))
            redundancy = 0.0
            for s in selected:
                sim = sum(x * y for x, y in zip(c["_vec"], s["_vec"]))
                redundancy = max(redundancy, sim)
            score = lambda_ * relevance - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        pool.remove(best)
    return [{"text": s["text"], "score": s.get("score", 0), "meta": s["meta"]}
            for s in selected]


def keyword_overlap_rerank(query, candidates, top_k=5, score_field=None):
    """
    DEPRECATED -- kept only for reference. Stages 4-7 now use the dedicated
    reranker module (common/reranker.py), which provides a real cross-encoder
    with an IDF-weighted heuristic fallback and score thresholding. This plain
    overlap version is what the project started with; compare it against
    reranker.IDFHeuristicReranker to see why IDF weighting matters.
    """
    q_terms = _tok(query)
    q_set = set(q_terms)
    n = len(candidates)
    reranked = []
    for prior_rank, c in enumerate(candidates):
        target = c["text"]
        if score_field and c.get("meta", {}).get(score_field):
            target = c["meta"][score_field]
        d_terms = _tok(target)
        d_set = set(d_terms)
        overlap = len(q_set & d_set)
        coverage = overlap / (len(q_set) + 1e-9)
        # phrase bonus: reward consecutive query-term matches
        phrase = 0
        for i in range(len(d_terms) - 1):
            if d_terms[i] in q_set and d_terms[i + 1] in q_set:
                phrase += 1
        # The cross-encoder signal (joint coverage + phrase adjacency) is the
        # main term, but we BLEND IN the prior retrieval rank rather than
        # discarding it. A real reranker is used the same way: as a re-score
        # on top of a strong first-stage ranking, not a blind replacement.
        joint_signal = coverage * 2.0 + phrase * 0.5
        prior_signal = (n - prior_rank) / n  # 1.0 for the current top hit
        rerank_score = 0.75 * joint_signal + 0.25 * prior_signal
        reranked.append({**c, "rerank_score": rerank_score})
    reranked.sort(key=lambda x: -x["rerank_score"])
    return reranked[:top_k]


# --------------------------------------------------------------------------
# PROMPT ASSEMBLY
# --------------------------------------------------------------------------

def build_prompt(question, chunks, lost_in_middle=False):
    """
    Assemble the final prompt. If lost_in_middle is True, the most relevant
    chunks are placed at the start and end, weakest in the middle.
    """
    texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
    if lost_in_middle and len(texts) > 2:
        ordered, left, right = [], [], []
        for i, t in enumerate(texts):
            (left if i % 2 == 0 else right).append(t)
        ordered = left + list(reversed(right))
        texts = ordered
    context = "\n\n---\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    return (
        "Answer the question using only the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
