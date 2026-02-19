from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import faiss

@dataclass
class DocChunk:
    chunk_id: str
    text: str
    source: str
    file_id: str

class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._indexes: Dict[str, faiss.IndexFlatIP] = {}
        self._chunks: Dict[str, List[DocChunk]] = {}
        self._vectors: Dict[str, np.ndarray] = {}

    def _get_index(self, file_id: str) -> faiss.IndexFlatIP:
        if file_id not in self._indexes:
            self._indexes[file_id] = faiss.IndexFlatIP(self.dim)
            self._chunks[file_id] = []
            self._vectors[file_id] = np.zeros((0, self.dim), dtype = np.float32)
        return self._indexes[file_id]
    
    def upsert(self, file_id: str, vectors: List[List[float]], chunks: List[DocChunk]):
        index = self._get_index(file_id)
        
        vec = np.array(vectors, dtype = np.float32)
        faiss.normalize_L2(vec)

        index.add(vec)
        self._chunks[file_id].extend(chunks)
        self._vectors[file_id] = np.vstack([self._vectors[file_id], vec])

    def search(self, file_id: str, query_vector: List[float], k: int = 6) -> List[Tuple[DocChunk, float]]:
        if file_id not in self._indexes:
            return []
        
        query = np.array([query_vector], dtype = np.float32)
        faiss.normalize_L2(query)

        index = self._indexes[file_id]
        scores, indices = index.search(query, k)

        results: List[Tuple[DocChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            chunk = self._chunks[file_id][int(idx)]
            results.append((chunk, float(score)))
        return results