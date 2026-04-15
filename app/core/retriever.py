import os
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi


class Retriever:
    _instance: Optional["Retriever"] = None

    @classmethod
    def get_instance(cls) -> "Retriever":
        """Singleton — keeps BM25 index alive across requests."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

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

        # In-memory mode: no Qdrant server needed
        if os.getenv("QDRANT_MODE", "server") == "memory":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        self._ensure_collection()

        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[str] = []

        # Rebuild BM25 index from existing Qdrant data on startup
        self._rebuild_bm25_from_qdrant()

    def _ensure_collection(self, vector_size: int = None):
        size = vector_size or int(os.getenv("EMBED_DIM", "768"))
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=size,
                    distance=Distance.COSINE
                )
            )

    def _generate_id(self, text: str) -> str:
        return str(uuid.uuid4())

    def add_documents(self, chunks: List[str], vectors: List[List[float]]):
        points = []
        self.chunks.extend(chunks)
        
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
        
        self._build_bm25_index(self.chunks)

    def _build_bm25_index(self, chunks: List[str]):
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)

    def _rebuild_bm25_from_qdrant(self):
        """Rebuild BM25 index from chunks already stored in Qdrant."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            if total_points == 0:
                return

            # Scroll all points to get their text payloads
            all_chunks = []
            offset = None
            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in results:
                    text = point.payload.get("text", "")
                    idx = point.payload.get("chunk_index", len(all_chunks))
                    all_chunks.append((idx, text))

                if next_offset is None:
                    break
                offset = next_offset

            # Sort by chunk_index to maintain order
            all_chunks.sort(key=lambda x: x[0])
            self.chunks = [text for _, text in all_chunks]
            self._build_bm25_index(self.chunks)
        except Exception:
            # Qdrant not available yet (e.g., during testing) — skip
            pass

    def dense_retrieval(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        k = top_k or self.top_k

        try:
            # qdrant-client >= 1.12
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k,
                with_payload=True,
            )
            points = results.points
        except AttributeError:
            # qdrant-client < 1.12
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
            )

        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "chunk_index": hit.payload.get("chunk_index"),
                "method": "dense"
            }
            for hit in points
        ]

    def sparse_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25_index:
            return []
        
        k = top_k or self.top_k
        
        scores = self.bm25_index.get_scores(query.lower().split())
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
