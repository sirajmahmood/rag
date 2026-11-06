"""
eval/harness.py
---------------
A reusable evaluation harness. Every stage exposes a `RAGPipeline` class with
a `.query(question)` method that returns:

    {
      "answer":  str,
      "contexts": [str, ...]   # the retrieved chunks used for generation
    }

The harness runs the pipeline over the golden set and computes the standard
RAG metrics:

  Context Recall    -- did retrieval surface the chunk(s) with the answer?
  Context Precision -- what fraction of retrieved chunks were on-topic?
  Answer Coverage   -- does the final answer contain the expected facts?
  Faithfulness      -- is the answer grounded in the retrieved context?

Run a single stage:   python eval/harness.py stages.stage_1_naive
Run ALL stages:       python eval/harness.py            (the comparison table)
"""

import re
import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.dataset import GOLD  # noqa: E402


def _tok(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _context_recall(contexts, gold_keywords):
    """Fraction of gold keywords found anywhere in the retrieved contexts."""
    joined = _tok(" ".join(contexts))
    hits = sum(1 for kw in gold_keywords if kw.lower() in joined)
    return hits / max(1, len(gold_keywords))


def _context_precision(contexts, key_terms):
    """
    Precision: what fraction of retrieved chunks are genuinely on-topic.

    A chunk counts as on-topic if it contains any of the question's `key_terms`
    -- the distinctive words specific to the answer. This is size-neutral: a
    focused 70-word child chunk and a 110-word flat chunk are judged the same
    way, so the metric compares parent-child retrieval against flat retrieval
    fairly, instead of rewarding whichever chunking packs in more keywords.
    """
    if not contexts:
        return 0.0
    keys = {k.lower() for k in key_terms}
    if not keys:
        return 0.0
    on_topic = sum(1 for c in contexts if any(k in c.lower() for k in keys))
    return on_topic / len(contexts)


def _answer_coverage(answer, expected_terms):
    """Fraction of expected fact-terms present in the answer."""
    a = answer.lower()
    hits = sum(1 for t in expected_terms if t.lower() in a)
    return hits / max(1, len(expected_terms))


def _faithfulness(answer, contexts):
    """
    Approximate groundedness: fraction of content words in the answer that
    also appear in the retrieved context. Low score => possible hallucination.
    """
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are",
            "that", "this", "it", "for", "on", "with", "as", "by", "be",
            "from", "at", "which", "can", "not", "i", "do", "have", "they"}
    ans = [w for w in _tok(answer) if w not in stop and len(w) > 2]
    if not ans:
        return 0.0
    ctx = _tok(" ".join(contexts))
    grounded = sum(1 for w in ans if w in ctx)
    return grounded / len(ans)


def evaluate(pipeline, verbose=False):
    rows = []
    for item in GOLD:
        result = pipeline.query(item["question"])
        answer = result.get("answer", "")
        contexts = result.get("contexts", [])
        # For parent-child stages, retrieval quality is judged on the precise
        # CHILD chunk that was matched, not the large parent fed to the LLM.
        # Stages without parent-child simply omit retrieval_units.
        units = result.get("retrieval_units", contexts)
        row = {
            "question": item["question"],
            # Recall: did the answer-bearing content reach the LLM at all?
            # Measured on the context actually delivered for generation.
            "recall": _context_recall(contexts, item["gold_keywords"]),
            # Precision: how focused was retrieval? Measured on the precise
            # unit that was matched (the child chunk in parent-child stages).
            "precision": _context_precision(units, item["key_terms"]),
            "coverage": _answer_coverage(answer, item["expected_terms"]),
            "faithfulness": _faithfulness(answer, contexts),
        }
        rows.append(row)
        if verbose:
            print(f"\nQ: {item['question']}")
            print(f"  recall={row['recall']:.2f} precision={row['precision']:.2f} "
                  f"coverage={row['coverage']:.2f} faithfulness={row['faithfulness']:.2f}")
            print(f"  A: {answer[:160]}")

    n = len(rows)
    agg = {
        "recall": sum(r["recall"] for r in rows) / n,
        "precision": sum(r["precision"] for r in rows) / n,
        "coverage": sum(r["coverage"] for r in rows) / n,
        "faithfulness": sum(r["faithfulness"] for r in rows) / n,
    }
    agg["overall"] = sum(agg.values()) / 4
    return agg, rows


STAGES = [
    ("stages.stage_1_naive",        "1. Naive RAG"),
    ("stages.stage_2_chunking",     "2. + Sentence chunking"),
    ("stages.stage_3_hybrid",       "3. + Hybrid retrieval (RRF)"),
    ("stages.stage_4_rerank",       "4. + Reranking"),
    ("stages.stage_5_query",        "5. + Query rewriting / HyDE"),
    ("stages.stage_6_contextual",   "6. + Contextual ingest + MMR"),
    ("stages.stage_7_agentic",      "7. Agentic RAG loop"),
]


def run_one(module_name, verbose=True):
    mod = importlib.import_module(module_name)
    pipeline = mod.RAGPipeline()
    print(f"\n=== Evaluating {module_name}  (LLM: {pipeline.llm.name}) ===")
    agg, rows = evaluate(pipeline, verbose=verbose)
    print(f"\n--- Aggregate ---")
    for k in ["recall", "precision", "coverage", "faithfulness", "overall"]:
        print(f"  {k:14s}: {agg[k]:.3f}")
    return agg


def run_all():
    print("\nRunning every stage on the golden set...\n")
    results = []
    for module_name, label in STAGES:
        try:
            mod = importlib.import_module(module_name)
            pipeline = mod.RAGPipeline()
            agg, _ = evaluate(pipeline, verbose=False)
            results.append((label, agg))
        except Exception as e:
            print(f"  !! {label} failed: {e}")
            results.append((label, None))

    # comparison table
    print("\n" + "=" * 86)
    print("RAG STAGE COMPARISON  (all metrics 0.0 - 1.0, higher is better)")
    print("=" * 86)
    header = f"{'Stage':<32}{'Recall':>10}{'Precision':>11}{'Coverage':>10}{'Faith.':>9}{'OVERALL':>10}"
    print(header)
    print("-" * 86)
    prev = None
    for label, agg in results:
        if agg is None:
            print(f"{label:<32}{'ERROR':>50}")
            continue
        delta = ""
        if prev is not None:
            d = agg["overall"] - prev
            delta = f"  ({'+' if d >= 0 else ''}{d:.3f})"
        print(f"{label:<32}{agg['recall']:>10.3f}{agg['precision']:>11.3f}"
              f"{agg['coverage']:>10.3f}{agg['faithfulness']:>9.3f}"
              f"{agg['overall']:>10.3f}{delta}")
        prev = agg["overall"]
    print("=" * 86)
    print("Each row adds ONE technique on top of the previous stage.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
    else:
        run_all()
