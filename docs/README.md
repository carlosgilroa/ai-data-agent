# AI Data Agent Documentation Hub

Welcome to the technical documentation for the **AI-Native Semantic Query Agent & RAG Microservice**.

---

## Documentation Structure

```text
docs/
├── README.md                          # Documentation overview and sitemap (this file)
├── api/                               # REST API specifications and contracts
│   └── endpoints_reference.md         # Complete REST endpoints, schemas, and examples
├── architecture/                      # System design, RAG pipeline & safety guards
│   ├── agent_architecture.md          # Autonomous ReAct loop, session memory & SQL safety
│   └── semantic_rag_indexer.md        # LookML AST parser, views/models/joins & ChromaDB RAG
├── evaluation/                        # BI ground-truth verification & semantic alignment
│   └── looker_dashboard_reconciliation.md # [Active] Real-world reconciliation against production Looker Dashboard
└── deployment/                        # Local and production deployment guides (Upcoming)
    ├── local_development.md           # [Planned] Local venv, Windows execution, environment
    └── docker_compose.md              # [Planned] Containerization & Raspberry Pi integration
```

---

## Active Documentation

- **[REST API Endpoints Reference](./api/endpoints_reference.md)**: Full specification of all `/api/v1` endpoints (`/health`, `/catalog`, `/agent/query`, `/agent/session/{session_id}`, `/agent/sql`), request/response contracts, and code samples.
- **[Agent Architecture & ReAct Orchestration](./architecture/agent_architecture.md)**: Autonomous ReAct loop, minimalist zero-shot prompt design, tool binding, multi-turn sliding window memory, and AST execution guardrails.
- **[Semantic RAG Indexer Architecture](./architecture/semantic_rag_indexer.md)**: LookML single source of truth, AST extraction of views/models/joins (`lkml`), ChromaDB vector store, and data catalog retrieval.

---

## Core System Overview

The microservice connects verified **LookML semantic definitions** (from [`projects/wallet-lookml`](file:///c:/projects/wallet-lookml)) directly to local columnar Parquet lakehouse storage (from [`projects/wallet`](file:///c:/projects/wallet)), enabling accurate natural language text-to-SQL analytics without hallucinations or metric drift.
