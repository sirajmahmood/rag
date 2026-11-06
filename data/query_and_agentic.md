# Query Transformation and Agentic RAG

## Query rewriting

User queries are often short, ambiguous, or phrased differently from the source documents.
Query rewriting transforms the raw query before retrieval to improve the match. Techniques
include expanding the query with synonyms, rephrasing it more formally, and decomposing a
complex question into simpler sub-questions that are retrieved independently.

## HyDE

HyDE stands for Hypothetical Document Embeddings. Instead of embedding the question, the
system first asks the language model to write a hypothetical answer to the question, then
embeds that hypothetical answer and retrieves with it. The hypothetical answer is written
in the same style as real documents, so its embedding lands closer to the relevant chunks
than the short question would.

The reason HyDE improves retrieval is a mismatch of form. A user question and a source
passage are written differently: the question is short and interrogative, the passage is
long and declarative. Their embeddings therefore sit far apart even when the passage holds
the answer. A hypothetical answer written by the language model has the declarative form
of a real passage, so embedding it closes that gap. HyDE works best for questions phrased
very differently from the documents, and it adds one extra language model call per query.

A practical caveat with HyDE: if the hypothetical answer invents wrong details, its
embedding can drift toward irrelevant chunks. For this reason HyDE is often combined with
ordinary query embedding, retrieving with both and fusing the results so a bad
hypothetical answer cannot derail retrieval on its own.

## Multi-query retrieval

Multi-query retrieval generates several paraphrases of the original question, retrieves
chunks for each paraphrase, and merges the deduplicated results. This broadens recall and
reduces the chance that a single awkward phrasing misses the relevant document.

## Contextual ingestion

Contextual ingestion improves chunks at indexing time. Before storing a chunk, a language
model writes a short description of how the chunk fits into its source document, and that
description is prepended to the chunk. This prevents chunks from being decontextualized,
which is a common cause of retrieval failure when a chunk uses pronouns or references that
only make sense within the full document.

## Agentic RAG

Agentic RAG replaces single-shot retrieval with an iterative loop driven by the model's
own reasoning. The agent plans which sub-questions to ask, retrieves for each, and judges
whether the retrieved context is sufficient. If the context is insufficient, the agent
reformulates the query and retrieves again. This loop continues until the agent has enough
evidence to answer or reaches a step limit.

## Corrective RAG

Corrective RAG adds a grading step after retrieval. A grader checks whether the retrieved
chunks are relevant to the question. If relevance is low, the system can fall back to a
different source, such as web search, or trigger a query rewrite before answering.

## Self-RAG

Self-RAG trains the model to decide for itself when retrieval is needed and to critique
its own output using reflection tokens that mark whether a statement is supported by the
retrieved evidence. This makes retrieval adaptive rather than always-on.
