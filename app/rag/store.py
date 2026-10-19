import os
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.schemas.catalog import CatalogItem

logger = logging.getLogger(__name__)

# In-memory fallback if chromadb is not yet installed in runtime
_MEMORY_STORE: Dict[str, CatalogItem] = {}


class VectorStoreManager:
    """Manages ChromaDB vector embeddings for LookML metrics and schema catalog."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.client = None
        self.collection = None
        self.chroma_available = False
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="semantic_catalog",
                metadata={"description": "LookML Semantic Metrics & Data Lake Dimensions"}
            )
            self.chroma_available = True
            logger.info("ChromaDB persistent vector store initialized at %s", self.persist_dir)
        except ImportError:
            logger.warning("chromadb not installed in current environment. Using in-memory catalog fallback.")
            self.chroma_available = False
        except Exception as e:
            logger.error("Failed to initialize ChromaDB: %s. Using in-memory catalog fallback.", e)
            self.chroma_available = False

    def add_items(self, items: List[CatalogItem]):
        """Upsert catalog items into vector store or fallback memory."""
        if not items:
            return

        if self.chroma_available and self.collection is not None:
            ids = [item.id for item in items]
            documents = [
                f"Name: {item.name}\nType: {item.item_type}\nView: {item.view}\nDescription: {item.description}\nSQL: {item.sql_formula or 'N/A'}\nExamples: {', '.join(item.examples)}"
                for item in items
            ]
            metadatas = [
                {
                    "item_type": item.item_type,
                    "name": item.name,
                    "view": item.view,
                    "sql_formula": item.sql_formula or "",
                    "table_name": item.table_name or "",
                    "description": item.description,
                }
                for item in items
            ]
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info("Upserted %d items into ChromaDB collection", len(items))

        # Always update memory cache
        for item in items:
            _MEMORY_STORE[item.id] = item

    def search(self, query: str, top_k: int = 5) -> List[CatalogItem]:
        """Search catalog for matching metrics or dimensions using semantic similarity or keyword matching."""
        if self.chroma_available and self.collection is not None:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k, max(1, self.collection.count()))
                )
                matching_items: List[CatalogItem] = []
                if results and "metadatas" in results and results["metadatas"]:
                    for idx, meta in enumerate(results["metadatas"][0]):
                        doc_id = results["ids"][0][idx]
                        matching_items.append(
                            CatalogItem(
                                id=doc_id,
                                item_type=meta.get("item_type", "metric"),
                                name=meta.get("name", ""),
                                view=meta.get("view", ""),
                                description=meta.get("description", ""),
                                sql_formula=meta.get("sql_formula"),
                                table_name=meta.get("table_name"),
                                examples=[]
                            )
                        )
                return matching_items
            except Exception as e:
                logger.error("ChromaDB search failed: %s. Falling back to memory search.", e)

        # Fallback keyword match
        tokens = query.lower().split()
        scored: List[tuple[int, CatalogItem]] = []
        for item in _MEMORY_STORE.values():
            text = f"{item.name} {item.description} {item.view} {item.item_type}".lower()
            score = sum(1 for token in tokens if token in text)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]] or list(_MEMORY_STORE.values())[:top_k]

    def list_all(self) -> List[CatalogItem]:
        """Return all catalog items."""
        return list(_MEMORY_STORE.values())

    def count(self) -> int:
        if self.chroma_available and self.collection is not None:
            try:
                return self.collection.count()
            except Exception:
                pass
        return len(_MEMORY_STORE)


# Global singleton instance
vector_store = VectorStoreManager()
