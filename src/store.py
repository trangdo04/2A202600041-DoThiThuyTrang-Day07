from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0
        
        # Bypass ChromaDB to avoid silent C++ DLL crashes on this Windows runtime
        self._use_chroma = False
        self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata if doc.metadata else {}
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        q_emb = self._embedding_fn(query)
        results = []
        for r in records:
            sim = compute_similarity(q_emb, r['embedding'])
            out = r.copy()
            out['score'] = sim
            results.append(out)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        for doc in docs:
            record = self._make_record(doc)
            emb = self._embedding_fn(doc.content)
            record['embedding'] = emb
            self._store.append(record)
            
            ids.append(doc.id)
            documents.append(doc.content)
            embeddings.append(emb)
            metadatas.append(record['metadata'])
            
        if self._use_chroma and self._collection is not None:
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k)
            
        filtered = []
        for r in self._store:
            match = True
            for k, v in metadata_filter.items():
                if r['metadata'].get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(r)
                
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        initial_len = len(self._store)
        new_store = []
        for r in self._store:
            if r.get('id') == doc_id or r.get('metadata', {}).get('doc_id') == doc_id:
                pass
            else:
                new_store.append(r)
        
        removed = (len(new_store) < initial_len)
        self._store = new_store
        
        if self._use_chroma and self._collection is not None and removed:
            try:
                self._collection.delete(ids=[doc_id])
            except Exception:
                pass
                
        return removed
