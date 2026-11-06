# Retrieval-Augmented Generation Fundamentals

Retrieval-Augmented Generation (RAG) is a technique that combines a large language
model with an external knowledge source. Instead of relying only on the parameters
learned during training, a RAG system retrieves relevant documents at query time and
feeds them into the model's context window.

## Why RAG exists

Large language models have a fixed knowledge cutoff and cannot know about events or
documents created after training. They also tend to hallucinate facts when asked about
information they never saw. RAG addresses both problems by grounding the model's answer
in retrieved text, so the model can cite real sources instead of inventing them.

## The core pipeline

A naive RAG pipeline has three steps. First, ingestion: documents are split into chunks
and each chunk is converted into a vector embedding, then stored in a vector database.
Second, retrieval: the user query is embedded and the most similar chunks are fetched
using nearest-neighbor search. Third, generation: the retrieved chunks are inserted into
a prompt and the language model produces an answer grounded in that context.

## Embeddings

An embedding is a dense numerical vector that captures the semantic meaning of a piece of
text. Texts with similar meaning have vectors that are close together in vector space,
measured by cosine similarity. The quality of embeddings directly determines retrieval
quality, which is why choosing a strong embedding model matters.

## Chunk size tradeoffs

Small chunks give precise retrieval but may lack surrounding context. Large chunks carry
more context but dilute the embedding with multiple topics, hurting retrieval precision.
A common compromise is to retrieve small chunks but return their larger parent chunk to
the language model. Overlap between chunks prevents important sentences from being split.
