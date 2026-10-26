import os
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.schemas.catalog import CatalogItem

logger = logging.getLogger(__name__)

# In-memory cache of indexed items
_MEMORY_STORE: Dict[str, CatalogItem] = {}

COLLECTION_NAME = "semantic_catalog"
COLLECTION_METADATA = {
    "description": "LookML Semantic Metrics & Data Lake Dimensions",
    "hnsw:space": "cosine",
}


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

            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            try:
                self.collection = self.client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata=COLLECTION_METADATA,
                )
                # If existing on-disk collection was created before cosine space was configured, recreate it
                if (self.collection.metadata or {}).get("hnsw:space") != "cosine":
                    logger.info("Upgrading ChromaDB collection '%s' to cosine distance space.", COLLECTION_NAME)
                    self.client.delete_collection(name=COLLECTION_NAME)
                    self.collection = self.client.create_collection(
                        name=COLLECTION_NAME,
                        metadata=COLLECTION_METADATA,
                    )
            except Exception:
                # Handle metadata mismatch on existing collection by recreating with cosine space
                try:
                    self.client.delete_collection(name=COLLECTION_NAME)
                except Exception:
                    pass
                self.collection = self.client.create_collection(
                    name=COLLECTION_NAME,
                    metadata=COLLECTION_METADATA,
                )
            self.chroma_available = True
            logger.info("ChromaDB persistent vector store initialized at %s (space=cosine)", self.persist_dir)
        except ImportError:
            logger.warning("chromadb not installed in current environment. Using in-memory catalog fallback.")
            self.chroma_available = False
        except Exception as e:
            logger.error("Failed to initialize ChromaDB: %s. Using in-memory catalog fallback.", e)
            self.chroma_available = False

    def clear(self):
        """Wipe all items from the ChromaDB collection and in-memory store before fresh indexing."""
        _MEMORY_STORE.clear()
        if self.chroma_available and self.client is not None:
            try:
                self.client.delete_collection(name=COLLECTION_NAME)
            except Exception as e:
                logger.debug("Collection delete skipped or already absent: %s", e)
            try:
                self.collection = self.client.create_collection(
                    name=COLLECTION_NAME,
                    metadata=COLLECTION_METADATA,
                )
                logger.info("Cleared and recreated ChromaDB collection '%s'", COLLECTION_NAME)
            except Exception as e:
                logger.error("Failed to recreate ChromaDB collection after clear: %s", e)

    def add_items(self, items: List[CatalogItem]):
        """Upsert catalog items into vector store and memory cache."""
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
                    "explores": ",".join(item.explores),
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

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        item_types: Optional[List[str]] = None,
    ) -> List[CatalogItem]:
        """Search catalog for matching metrics or dimensions using dynamic relative similarity filtering.

        item_types limits the search to some catalog entry types (e.g. explores, metrics, dimensions).
        """
        effective_top_k = top_k if top_k is not None else settings.RAG_TOP_K
        noise_floor = settings.RAG_MIN_SIMILARITY if min_similarity is None else min_similarity
        rel_ratio = settings.RAG_RELATIVE_CUTOFF_RATIO

        if self.chroma_available and self.collection is not None:
            try:
                total_count = self.collection.count()
                if total_count == 0:
                    return []

                # Query candidate pool
                cand_limit = min(max(effective_top_k * 2, 10), total_count)
                query_kwargs: Dict[str, Any] = {}
                if item_types:
                    query_kwargs["where"] = {"item_type": {"$in": list(item_types)}}
                results = self.collection.query(
                    query_texts=[query],
                    n_results=cand_limit,
                    include=["metadatas", "distances"],
                    **query_kwargs,
                )
                metadatas_list = (results or {}).get("metadatas") or [[]]
                distances_list = (results or {}).get("distances") or [[]]
                ids_list = (results or {}).get("ids") or [[]]

                candidates: List[CatalogItem] = []
                if metadatas_list and metadatas_list[0]:
                    for idx, meta in enumerate(metadatas_list[0]):
                        doc_id = ids_list[0][idx]
                        raw_dist = float(distances_list[0][idx]) if (distances_list and distances_list[0] and idx < len(distances_list[0])) else 0.0
                        dist_val = round(raw_dist, 4)
                        sim_val = round(max(0.0, 1.0 - raw_dist), 4)

                        candidates.append(
                            CatalogItem(
                                id=doc_id,
                                item_type=meta.get("item_type", "metric"),
                                name=meta.get("name", ""),
                                view=meta.get("view", ""),
                                description=meta.get("description", ""),
                                sql_formula=meta.get("sql_formula") or None,
                                table_name=meta.get("table_name") or None,
                                examples=[],
                                distance=dist_val,
                                similarity_score=sim_val,
                                explores=[e for e in (meta.get("explores") or "").split(",") if e],
                            )
                        )

                if not candidates:
                    return []

                # Determine top score and evaluate against noise floor
                top_score = candidates[0].similarity_score or 0.0
                if top_score < noise_floor:
                    logger.debug(
                        "Query '%s' rejected: top similarity score %.4f is below noise floor %.4f",
                        query, top_score, noise_floor
                    )
                    return []

                # Dynamic relative cutoff: retain candidates within rel_ratio (80%) of top score
                dynamic_cutoff = max(noise_floor, round(top_score * rel_ratio, 4))
                matching_items = [
                    item for item in candidates
                    if item.similarity_score is not None and item.similarity_score >= dynamic_cutoff
                ]
                return matching_items[:effective_top_k]
            except Exception as e:
                logger.error("ChromaDB search failed: %s. Falling back to memory search.", e)

        # Fallback keyword match when ChromaDB is unavailable
        tokens = query.lower().split()
        if not tokens:
            return []
        scored: List[tuple[float, CatalogItem]] = []
        for item in _MEMORY_STORE.values():
            if item_types and item.item_type not in item_types:
                continue
            text = f"{item.name} {item.description} {item.view} {item.item_type}".lower()
            hits = sum(1 for token in tokens if token in text)
            if hits > 0:
                sim = round(min(1.0, hits / len(tokens)), 4)
                dist = round(1.0 - sim, 4)
                if sim >= noise_floor:
                    item_copy = item.model_copy(update={"similarity_score": sim, "distance": dist})
                    scored.append((sim, item_copy))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sim = scored[0][0]
        dynamic_cutoff = max(noise_floor, round(top_sim * rel_ratio, 4))
        filtered = [item for sim, item in scored if sim >= dynamic_cutoff]
        return filtered[:effective_top_k]

    def get_by_id(self, doc_id: str) -> Optional[CatalogItem]:
        """Retrieve a specific catalog item by its ID (e.g. 'table.dim_accounts')."""
        if doc_id in _MEMORY_STORE:
            return _MEMORY_STORE[doc_id]
        if self.chroma_available and self.collection is not None:
            try:
                res = self.collection.get(ids=[doc_id], include=["metadatas"])
                metas = res.get("metadatas") or []
                if metas:
                    meta = metas[0]
                    return CatalogItem(
                        id=doc_id,
                        item_type=meta.get("item_type", "table"),
                        name=meta.get("name", ""),
                        view=meta.get("view", ""),
                        description=meta.get("description", ""),
                        sql_formula=meta.get("sql_formula") or None,
                        table_name=meta.get("table_name") or None,
                        examples=[],
                        explores=[e for e in (meta.get("explores") or "").split(",") if e],
                    )
            except Exception:
                pass
        return None

    def list_all(self) -> List[CatalogItem]:
        """Return all catalog items."""
        if _MEMORY_STORE:
            return list(_MEMORY_STORE.values())
        if self.chroma_available and self.collection is not None:
            try:
                res = self.collection.get(include=["metadatas"])
                items: List[CatalogItem] = []
                for idx, meta in enumerate(res.get("metadatas") or []):
                    doc_id = res["ids"][idx]
                    items.append(
                        CatalogItem(
                            id=doc_id,
                            item_type=meta.get("item_type", "metric"),
                            name=meta.get("name", ""),
                            view=meta.get("view", ""),
                            description=meta.get("description", ""),
                            sql_formula=meta.get("sql_formula") or None,
                            table_name=meta.get("table_name") or None,
                            examples=[],
                            explores=[e for e in (meta.get("explores") or "").split(",") if e],
                        )
                    )
                return items
            except Exception:
                pass
        return []

    def count(self) -> int:
        if self.chroma_available and self.collection is not None:
            try:
                return self.collection.count()
            except Exception:
                pass
        return len(_MEMORY_STORE)


# Global singleton instance
vector_store = VectorStoreManager()
