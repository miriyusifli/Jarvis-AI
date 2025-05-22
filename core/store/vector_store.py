from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    def add_texts(self, texts: list[str], metadatas: list[dict] = None) -> list[str]:
        """Add text documents to the vector store"""

    def similarity_search(self, query: str, k: int = 4) -> list[dict]:
        """Search for similar documents given a query"""

    def persist(self):
        """Persist the vector store to disk"""
