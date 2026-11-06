# Evaluating and Operating RAG Systems

## Why evaluation matters

A RAG system has many moving parts, and changing one part can silently break
another. Without measurement you cannot tell whether a change helped. A proper
evaluation harness runs the pipeline over a fixed set of questions and reports
metrics, so every change can be compared against a baseline.

## Retrieval metrics

Context recall measures whether the chunks containing the answer were actually
retrieved. Context precision measures what fraction of the retrieved chunks were
genuinely relevant, rather than noise. A system can have high recall but low
precision if it retrieves many chunks, most of them irrelevant.

## Answer metrics

Answer coverage checks whether the final answer contains the facts the question
asked for. Faithfulness, sometimes called groundedness, checks whether every
claim in the answer is supported by the retrieved context. A faithful answer
never introduces facts that are absent from the context, which is how RAG
reduces hallucination.

## The golden dataset

Evaluation needs a golden dataset of questions paired with known correct answers
and the documents that contain them. The dataset should be built from real user
queries when possible. Even two hundred well-chosen questions are enough to catch
most regressions when a retrieval change is made.

## Operating concerns

Latency matters in production. A two-stage retrieval design keeps latency low: a
fast approximate search produces a wide candidate set, and only that small set is
passed to the slower reranker. Caching answers to frequent queries reduces both
latency and cost. Monitoring retrieval precision over time catches silent
degradation as the document collection grows.
