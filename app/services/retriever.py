from app.services.bm25_index import BM25Index
from app.services.reranker import Reranker
from langchain_core.documents import Document

bm25_index = None
reranker = Reranker()


def reset_bm25():
    global bm25_index
    bm25_index = None


def hybrid_retrieve(query, vectorstore, k=3):
    global bm25_index

    if vectorstore is None:
        return []

    # ✅ smaller retrieval
    vector_results = vectorstore.similarity_search(query, k=3)

    # Lazy BM25 init
    if bm25_index is None:
        data = vectorstore.get()

        docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]

        bm25_index = BM25Index(docs)

    bm25_results = bm25_index.search(query, k=2)

    # ✅ dedupe fast
    seen = set()
    combined = []
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    # ✅ ONLY rerank if needed
    if len(combined) > 3:
        combined = reranker.rerank(query, combined, top_k=3)

    return combined[:3]