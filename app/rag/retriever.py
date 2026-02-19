from typing import List
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
from app.rag.store import FaissStore, DocChunk

class RAGIndex:
    def __init__(self):
        self._emb = OpenAIEmbeddings(
            model = settings.embedding_model,
            openai_api_key = settings.openai_api_key
        )
        self._store: FaissStore | None = None

    async def _ensure_store(self):
        if self._store is None:
            v = self._emb.embed_query("dimension probe")
            self._store = FaissStore(dim = len(v))

    async def ingest(self, file_id: str, source_url: str, chunks: List[str]) -> int:
        await self._ensure_store()
        assert self._store is not None

        vectors = self._emb.embed_documents(chunks)
        doc_chunks = [
            DocChunk(
                chunk_id = f"{file_id}:{i}",
                text = chunks[i],
                source = source_url,
                file_id = file_id
            )
            for i in range(len(chunks))
        ]

        self._store.upsert(file_id, vectors, doc_chunks)
        return len(chunks)
    
    async def retrieve(self, file_id: str, query: str, k: int = 6) -> list[tuple[DocChunk, float]]:
        await self._ensure_store()
        assert self._store is not None

        query_vec = self._emb.embed_query(query)
        results = self._store.search(file_id, query_vec, k = k)
        return results