"""Standalone CLI script to build and verify ChromaDB semantic index from LookML and Parquets."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.indexer import indexer
from app.rag.store import vector_store


def main():
    print("==================================================")
    print("   AI Data Agent: Semantic Catalog Indexer")
    print("==================================================")

    res = indexer.run_indexing()
    print(f"\n[+] Successfully indexed {res['total_indexed']} semantic items:")
    print(f"    - Metrics / Measures: {res['metrics']}")
    print(f"    - Dimensions:         {res['dimensions']}")
    print(f"    - Lakehouse Tables:   {res['tables']}")

    print("\n[+] Testing sample semantic search queries:")
    sample_queries = [
        "What is my net cashflow and savings rate?",
        "How are total expenses calculated?",
        "What is Tier 1 operational living spend?",
        "Show transaction account classifications"
    ]

    for q in sample_queries:
        print(f"\n--- Search: '{q}' ---")
        matches = vector_store.search(q, top_k=2)
        for m in matches:
            formula = f" | SQL: {m.sql_formula}" if m.sql_formula else ""
            print(f"  * {m.name} [{m.item_type} in {m.view}]{formula}")
            print(f"    {m.description[:90]}...")


if __name__ == "__main__":
    main()
