"""
eval/dataset.py
---------------
The golden evaluation set. Each item has:

  question        : the user query
  expected_terms  : facts a correct ANSWER must contain (answer-coverage metric)
  gold_doc        : the data file that contains the answer
  gold_keywords   : terms that should appear somewhere across the retrieved
                    context (context-recall metric)
  key_terms       : the distinctive, on-topic terms. A retrieved chunk counts
                    as relevant for the precision metric if it contains any
                    key term. These are chosen to be specific to the answer,
                    not generic words -- which makes precision size-neutral:
                    it does not matter whether a chunk is 70 or 200 words, only
                    whether it is actually about the right thing.

This is intentionally small and hand-built. In a real project you would grow
it to a few hundred items, ideally sampled from real user queries.
"""

GOLD = [
    {
        "question": "What are the three steps of a naive RAG pipeline?",
        "expected_terms": ["ingestion", "retrieval", "generation"],
        "gold_doc": "rag_fundamentals.md",
        "gold_keywords": ["ingestion", "generation", "nearest-neighbor"],
        "key_terms": ["ingestion", "nearest-neighbor"],
    },
    {
        "question": "Why does sparse retrieval like BM25 beat embeddings on rare terms?",
        "expected_terms": ["keyword", "exact", "rare"],
        "gold_doc": "retrieval_techniques.md",
        "gold_keywords": ["bm25", "keyword", "rare", "identifiers"],
        "key_terms": ["bm25", "identifiers"],
    },
    {
        "question": "How does Reciprocal Rank Fusion merge results without score normalization?",
        "expected_terms": ["rank", "normalization"],
        "gold_doc": "retrieval_techniques.md",
        "gold_keywords": ["reciprocal", "rank", "fusion", "normalization"],
        "key_terms": ["reciprocal", "normalization"],
    },
    {
        "question": "What problem does a cross-encoder reranker solve?",
        "expected_terms": ["relevance", "accurate", "cross-encoder"],
        "gold_doc": "retrieval_techniques.md",
        "gold_keywords": ["reranker", "cross-encoder", "relevance", "candidate"],
        "key_terms": ["reranker", "cross-encoder"],
    },
    {
        "question": "What is HyDE and why does it improve retrieval?",
        "expected_terms": ["hypothetical", "embed"],
        "gold_doc": "query_and_agentic.md",
        "gold_keywords": ["hyde", "hypothetical", "embeddings"],
        "key_terms": ["hyde", "hypothetical"],
    },
    {
        "question": "What does contextual ingestion do to a chunk before storing it?",
        "expected_terms": ["description", "document", "prepend"],
        "gold_doc": "query_and_agentic.md",
        "gold_keywords": ["contextual", "ingestion", "description", "decontextualized"],
        "key_terms": ["contextual", "decontextualized"],
    },
    {
        "question": "How does an agentic RAG loop decide whether to retrieve again?",
        "expected_terms": ["sufficient", "reformulate", "loop"],
        "gold_doc": "query_and_agentic.md",
        "gold_keywords": ["agentic", "iterative", "sufficient", "reformulates"],
        "key_terms": ["agentic", "reformulates", "iterative"],
    },
    {
        "question": "Why are chunks placed at the start and end of the context window?",
        "expected_terms": ["middle", "attend", "beginning"],
        "gold_doc": "retrieval_techniques.md",
        "gold_keywords": ["lost", "middle", "attend", "beginning"],
        "key_terms": ["attend", "middle"],
    },
    {
        "question": "What is the tradeoff between small and large chunks?",
        "expected_terms": ["precise", "context"],
        "gold_doc": "rag_fundamentals.md",
        "gold_keywords": ["small", "large", "dilute", "overlap"],
        "key_terms": ["dilute", "overlap"],
    },
    {
        "question": "What does corrective RAG do when retrieved chunks are not relevant?",
        "expected_terms": ["grader", "relevance", "fall back"],
        "gold_doc": "query_and_agentic.md",
        "gold_keywords": ["corrective", "grader", "relevance", "web search"],
        "key_terms": ["corrective", "grader"],
    },
]
