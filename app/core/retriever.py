import os
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(
        self,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        collection_name: Optional[str] = None,
        top_k: int = 5
    ):
        self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "qdrant")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "documents")
        self.top_k = top_k or int(os.getenv("TOP_K", "5"))
        
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        self._ensure_collection()
        
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[str] = []

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )

    def _generate_id(self, text: str) -> str:
        return str(uuid.uuid4())

    def add_documents(self, chunks: List[str], vectors: List[List[float]]):
        points = []
        self.chunks = chunks
        
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "chunk_index": i
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        self._build_bm25_index(chunks)

    def _build_bm25_index(self, chunks: List[str]):
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)

    def dense_retrieval(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True
        )
        
        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "chunk_index": hit.payload.get("chunk_index"),
                "method": "dense"
            }
            for hit in results
        ]

    def sparse_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25_index:
            return []
        
        k = top_k or self.top_k
        
        scores = self.bm25_index.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [
            {
                "text": self.chunks[i],
                "score": float(scores[i]),
                "chunk_index": i,
                "method": "sparse"
            }
            for i in top_indices if i < len(self.chunks)
        ]

    def hybrid_retrieval(self, query_vector: List[float], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        
        dense_results = self.dense_retrieval(query_vector, top_k=k * 2)
        sparse_results = self.sparse_retrieval(query, top_k=k * 2)
        
        combined = self._reciprocal_rank_fusion([dense_results, sparse_results], k=k)
        
        return combined[:k]

    def _reciprocal_rank_fusion(
        self, 
        results_list: List[List[Dict]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        doc_scores = {}
        
        for results in results_list:
            for rank, result in enumerate(results):
                doc_key = result["text"]
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {
                        "text": doc_key,
                        "chunk_index": result.get("chunk_index"),
                        "score": 0.0,
                        "methods": []
                    }
                
                score = 1.0 / (k + rank + 1)
                doc_scores[doc_key]["score"] += score
                doc_scores[doc_key]["methods"].append(result.get("method"))
        
        fused = list(doc_scores.values())
        fused.sort(key=lambda x: x["score"], reverse=True)
        
        return fused

    def retrieve(self, query_vector: List[float], query: str, mode: str = "hybrid", top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        if mode == "dense":
            return self.dense_retrieval(query_vector, top_k=k)
        elif mode == "sparse":
            return self.sparse_retrieval(query, top_k=k)
        else:
            return self.hybrid_retrieval(query_vector, query, top_k=k)
