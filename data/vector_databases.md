# Vector Databases and Embedding Models

## What a vector database stores

A vector database stores embedding vectors alongside the original text and its
metadata. At query time it performs nearest-neighbor search to find the vectors
closest to the query vector. Unlike a keyword index, it matches on meaning.

## Approximate nearest neighbor search

Searching every vector exactly is slow when there are millions of them.
Approximate nearest neighbor algorithms, such as HNSW, trade a tiny amount of
accuracy for a very large speed gain by organizing vectors into a navigable
graph. This is what makes dense retrieval fast enough for production.

## Choosing an embedding model

The embedding model determines retrieval quality more than any other single
choice. General-purpose embedding models work reasonably well, but a model
fine-tuned on the target domain matches its vocabulary far better. Retrieval-
optimized models are trained specifically to place questions near their answers.

## Asymmetric embeddings

A question is short and a document is long. Asymmetric embedding models use one
encoder for queries and a different encoder for documents, which improves the
match between a brief question and a detailed passage.

## Matryoshka embeddings

Matryoshka embeddings are trained so the vector can be truncated to fewer
dimensions with only a small loss of quality. A system can store short truncated
vectors for a fast first pass and full vectors for precise reranking, saving
storage and search cost.

## Metadata filtering

Storing metadata such as source, date, and section lets the system filter the
candidate set before nearest-neighbor search. Filtering by metadata narrows the
search space, which improves precision and speed when the collection spans many
topics or time periods.
