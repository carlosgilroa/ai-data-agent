# AI Data Agent Documentation Hub

Welcome to the technical documentation for the **AI-Native Semantic Query Agent & RAG Microservice**.

---

## Documentation Structure

```text
docs/
├── README.md                      # Documentation overview and sitemap (this file)
├── api/                           # REST API specifications and contracts
│   └── endpoints_reference.md     # Complete REST endpoints, schemas, and examples
├── architecture/                  # System design, RAG pipeline & safety guards (Upcoming)
│   ├── system_design.md           # [Planned] Orchestration, tool calling & data flow
│   └── ast_sql_safety.md          # [Planned] Safe read-only DuckDB execution model
├── evaluation/                    # Golden benchmarks & reconciliation (Upcoming)
│   ├── golden_suite.md            # [Under Review] Benchmark questions & ground-truth validation
│   └── metrics_and_drift.md       # [Planned] LookML semantic drift detection
└── deployment/                    # Local and production deployment guides (Upcoming)
    ├── local_development.md       # [Planned] Local venv, Windows execution, environment
    └── docker_compose.md          # [Planned] Containerization & Raspberry Pi integration
```

---

## Active Documentation

- **[REST API Endpoints Reference](./api/endpoints_reference.md)**: Full specification of all `/api/v1` endpoints (`/health`, `/catalog`, `/agent/query`, `/agent/sql`), request/response contracts, and code samples.

---

## Core System Overview

The microservice connects verified **LookML semantic definitions** (from [`projects/wallet-lookml`](file:///c:/projects/wallet-lookml)) directly to local columnar Parquet lakehouse storage (from [`projects/wallet`](file:///c:/projects/wallet)), enabling accurate natural language text-to-SQL analytics without hallucinations or metric drift.
