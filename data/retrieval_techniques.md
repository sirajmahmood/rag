# Retrieval Techniques and Reranking

## Dense versus sparse retrieval

Dense retrieval uses embedding vectors and nearest-neighbor search to find semantically
similar text. It excels at matching paraphrases and concepts even when no words overlap.
Sparse retrieval, such as BM25, scores documents by exact keyword overlap weighted by
term frequency and inverse document frequency. Sparse retrieval excels at matching rare
terms, product codes, names, and exact identifiers that embeddings often miss.

## Hybrid retrieval

Hybrid retrieval runs both dense and sparse retrieval and merges their results. The most
common merging method is Reciprocal Rank Fusion, which combines ranked lists by summing
one divided by a constant plus rank. Reciprocal Rank Fusion needs no score normalization
because it only uses the rank position, making it robust across very different scoring
scales. Hybrid retrieval typically outperforms either method alone.

## Reranking

Retrieval with bi-encoders is fast but approximate, because the query and document are
embedded independently. A reranker uses a cross-encoder that reads the query and a
candidate document together, producing a much more accurate relevance score. The standard
pipeline retrieves a large candidate set, for example one hundred chunks, then reranks and
keeps only the top five for generation. Reranking is slower per document but only runs on
the small candidate set, so it stays affordable.

## Maximal Marginal Relevance

When several retrieved chunks are near-duplicates, the context window wastes space on
redundant text. Maximal Marginal Relevance selects chunks that are both relevant to the
query and diverse relative to chunks already selected, improving coverage of the answer.

## Lost in the middle

Language models attend most strongly to the beginning and end of their context window.
Information placed in the middle of a long context is more likely to be ignored. A good
RAG system orders the most relevant retrieved chunks at the start and end of the prompt.
