"""
common/reranker.py
------------------
The reranker used in Stage 4 onward.

A reranker takes the wide candidate set from first-stage retrieval and re-scores
each (query, chunk) pair *jointly*, then keeps only the true top-k. This is the
"precision pass" of two-stage retrieval.

This module provides two implementations behind one factory, `get_reranker()`:

  1. CrossEncoderReranker  -- a REAL cross-encoder (sentence-transformers).
     A cross-encoder feeds the query and document through one transformer
     together, so it models their interaction directly. This is the genuine
     article. Used automatically when the model weights are available.

  2. IDFHeuristicReranker  -- an offline fallback that needs no model.
     The naive version of this (plain keyword overlap) had a real flaw: a long
     document-header chunk stuffed with common words scored high on coincidental
     overlap and displaced the true answer. This version fixes that with:
       - IDF weighting: rare, discriminative query terms count for much more
         than common ones, so generic words stop dominating.
       - length normalization: a chunk is not rewarded just for being long.
       - proximity bonus: query terms appearing close together score higher.
     The result reorders candidates the way a real reranker would on this
     corpus, without fetching any weights.

Both expose the same interface:

    reranker.rerank(query, candidates, top_k=5, score_field=None) -> list

`candidates` are result dicts {text, score, meta}. `score_field`, if given,
makes the reranker judge meta[score_field] (e.g. the precise child chunk)
instead of the returned text.
"""

import os
import re
import math
from collections import Counter


def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _target_text(candidate, score_field):
    """The text the reranker should actually score for this candidate."""
    if score_field and candidate.get("meta", {}).get(score_field):
        return candidate["meta"][score_field]
    return candidate["text"]


# ===========================================================================
# 1. REAL CROSS-ENCODER
# ===========================================================================

class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder. The model reads each
    (query, chunk) pair jointly and outputs a single relevance logit.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2, the standard small
    reranker trained on the MS MARCO passage-ranking task.
    """

    kind = "cross-encoder (real model)"

    def __init__(self, model_name=None):
        from sentence_transformers import CrossEncoder

        model_name = model_name or os.environ.get(
            "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query, candidates, top_k=5, score_field=None, min_ratio=0.0):
        if not candidates:
            return []
        pairs = [(query, _target_text(c, score_field)) for c in candidates]
        scores = self.model.predict(pairs)
        scored = []
        for c, s in zip(candidates, scores):
            scored.append({**c, "rerank_score": float(s)})
        scored.sort(key=lambda x: -x["rerank_score"])
        scored = scored[:top_k]
        if min_ratio > 0 and scored:
            # cross-encoder scores can be negative; shift to a positive range
            # before applying the ratio cutoff so the comparison is meaningful
            lo = min(s["rerank_score"] for s in scored)
            shift = -lo + 1e-6 if lo < 0 else 0.0
            top = scored[0]["rerank_score"] + shift
            cutoff = top * min_ratio
            kept = [s for s in scored if s["rerank_score"] + shift >= cutoff]
            scored = kept or scored[:1]
        return scored


# ===========================================================================
# 2. IDF-WEIGHTED HEURISTIC FALLBACK
# ===========================================================================

class IDFHeuristicReranker:
    """
    A model-free reranker that behaves like a real one on this corpus.

    It is fitted on the chunk collection at construction time so it knows which
    terms are rare (discriminative) and which are common (noise). That single
    fact -- IDF weighting -- is what stops a keyword-stuffed header chunk from
    out-scoring the genuine answer chunk, which was the flaw in the earlier
    plain-overlap reranker.
    """

    kind = "IDF heuristic (offline fallback)"

    def __init__(self, corpus_texts):
        # fit IDF over every chunk in the index
        self.N = max(1, len(corpus_texts))
        df = Counter()
        for text in corpus_texts:
            for term in set(_tok(text)):
                df[term] += 1
        self.idf = {}
        for term, count in df.items():
            # smoothed idf; rare terms -> high idf, ubiquitous terms -> ~0
            self.idf[term] = math.log((self.N + 1) / (count + 0.5))
        self._default_idf = math.log((self.N + 1) / 0.5)

    def _idf(self, term):
        return self.idf.get(term, self._default_idf)

    def _score(self, query, text):
        q_terms = _tok(query)
        if not q_terms:
            return 0.0
        q_set = set(q_terms)
        d_terms = _tok(text)
        if not d_terms:
            return 0.0
        d_count = Counter(d_terms)

        # (a) IDF-weighted coverage: how much of the query's *informative*
        #     mass is present in the chunk. Common words contribute little.
        q_idf_total = sum(self._idf(t) for t in q_set)
        matched_idf = sum(self._idf(t) for t in q_set if t in d_count)
        coverage = matched_idf / (q_idf_total + 1e-9)

        # (b) proximity: find the shortest window in the chunk containing the
        #     most query terms. Terms that appear close together indicate the
        #     chunk is genuinely about the query, not just mentioning words.
        positions = [i for i, t in enumerate(d_terms) if t in q_set]
        proximity = 0.0
        if len(positions) >= 2:
            span = positions[-1] - positions[0] + 1
            distinct = len({d_terms[i] for i in positions})
            proximity = distinct / span  # dense cluster -> close to 1.0

        # (c) phrase adjacency: consecutive query terms, IDF-weighted
        phrase = 0.0
        for i in range(len(d_terms) - 1):
            if d_terms[i] in q_set and d_terms[i + 1] in q_set:
                phrase += (self._idf(d_terms[i]) + self._idf(d_terms[i + 1])) / 2

        # (d) length penalty: a long chunk has more chance of coincidental
        #     overlap, so normalize gently by length. This is what neutralizes
        #     the keyword-stuffed header-chunk problem.
        length_penalty = 1.0 / (1.0 + math.log(1 + len(d_terms) / 60))

        raw = (coverage * 3.0) + (proximity * 1.0) + (phrase * 0.15)
        return raw * length_penalty

    def rerank(self, query, candidates, top_k=5, score_field=None, min_ratio=0.0):
        """
        min_ratio: if > 0, drop any candidate whose score is below
        min_ratio * (top candidate's score). This lets the reranker return
        FEWER than top_k results when the corpus simply does not contain that
        many relevant chunks -- which is what a real reranker enables and is
        the main lever for precision. At least one result is always kept.
        """
        if not candidates:
            return []
        n = len(candidates)
        scored = []
        for prior_rank, c in enumerate(candidates):
            text = _target_text(c, score_field)
            joint = self._score(query, text)
            # blend with the prior first-stage rank: a reranker re-scores on
            # top of a strong retriever, it does not discard its signal.
            prior_signal = (n - prior_rank) / n
            rerank_score = 0.8 * joint + 0.2 * prior_signal
            scored.append({**c, "rerank_score": rerank_score})
        scored.sort(key=lambda x: -x["rerank_score"])
        scored = scored[:top_k]
        if min_ratio > 0 and scored:
            cutoff = scored[0]["rerank_score"] * min_ratio
            kept = [s for s in scored if s["rerank_score"] >= cutoff]
            scored = kept or scored[:1]
        return scored


# ===========================================================================
# FACTORY
# ===========================================================================

def get_reranker(corpus_texts, force=None):
    """
    Return the best available reranker.

    corpus_texts : every chunk text in the index (needed to fit the heuristic).
    force        : "cross" or "heuristic" to override auto-detection.

    Auto-detection: try to load the real cross-encoder; if its weights cannot
    be downloaded/loaded (offline environment), fall back to the IDF heuristic.
    Set RAG_RERANKER=heuristic to skip the model-load attempt entirely.
    """
    want = force or os.environ.get("RAG_RERANKER", "auto")

    if want in ("cross", "auto"):
        try:
            r = CrossEncoderReranker()
            # smoke-test so a network failure surfaces here, not mid-eval
            r.model.predict([("test query", "test document")])
            return r
        except Exception as e:
            if want == "cross":
                raise
            print(f"[reranker] real cross-encoder unavailable ({type(e).__name__}); "
                  f"using IDF heuristic fallback")

    return IDFHeuristicReranker(corpus_texts)
