# ContextualCache

**A Fault-Tolerant Adaptive Semantic Cache for LLM Serving**

ContextualCache is a production-ready semantic caching system that sits between your application and LLM providers (Ollama, Groq, OpenAI). It dramatically reduces latency and API costs by serving cached responses for semantically similar queries, with formal correctness guarantees.

---

## Architecture Overview

```
Client → Tauri App (Analytics Dashboard)
              ↕ HTTP API
         FastAPI Backend
              │
    ┌─────────┴─────────┐
    │  Two-Tier Lookup   │
    │  T1: Exact Hash    │ ← O(1), <1ms
    │  T2: FAISS HNSW    │ ← O(log n), 2-15ms
    └─────────┬──────────┘
              │ MISS
              ▼
         LLM Provider (Ollama/Groq/OpenAI)
              │
    ┌─────────┴──────────┐
    │ W-TinyLFU Admission │  ← Rejects one-hit wonders
    │ CostGDSF Eviction   │  ← Keeps expensive-to-recompute entries
    │ Conformal Thresholds│  ← Per-entry τ with P(error) ≤ ε
    │ Session Context     │  ← Multi-turn EMA fusion
    │ Thompson Sampling   │  ← Adaptive threshold learning
    └────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Two-Tier Lookup** | Exact hash (T1, <1ms) + FAISS HNSW semantic search (T2) |
| **Per-Entry Conformal Thresholds** | Formal guarantee: P(incorrect hit) ≤ ε |
| **Semantic-W-TinyLFU Admission** | Rejects one-hit wonders via LSH-bucketed CMS |
| **CostGDSF Eviction** | Prioritizes retention of expensive-to-regenerate entries |
| **Session Context Fusion** | Multi-turn EMA with O(d) space per session |
| **Thompson Sampling Bandit** | Adaptive threshold prior learning with ADWIN drift detection |
| **Circuit Breakers** | Per-dependency fault isolation |
| **Premium Analytics Dashboard** | Real-time Tauri + SvelteKit + Chart.js |

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**
- **Ollama** (for local LLM) — or an API key for Groq/OpenAI
- **Rust** (for Tauri build, optional for dev mode)

### 1. Install Python Backend

```bash
cd sem-cache-c2

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure (Optional)

Create a `.env` file in the project root to override defaults:

```env
# LLM Configuration
CC_LLM_BACKEND=ollama          # ollama | groq | openai
CC_LLM_MODEL=llama3.2          # model name
CC_LLM_BASE_URL=http://localhost:11434

# For Groq/OpenAI:
# CC_LLM_API_KEY=your-api-key

# Cache Settings
CC_CACHE_CAPACITY=50000
CC_TARGET_ERROR_RATE=0.05      # conformal ε
CC_CONTEXT_ALPHA=0.85          # session fusion weight

# Embedding
CC_EMBEDDING_MODEL=all-MiniLM-L6-v2
CC_EMBEDDING_DIM=384
```

### 3. Start the Backend Server

```bash
python -m contextual_cache.main
```

The server starts on `http://localhost:8000`. You'll see:

```
============================================================
  ContextualCache v1.0 — Starting …
============================================================
  LLM backend:    ollama (llama3.2)
  Embedding:      all-MiniLM-L6-v2 (384d)
  Cache capacity:  50,000 entries
  Error rate (ε):  0.05
============================================================
```

### 4. Start the Tauri Frontend

```bash
cd sm-cache-tauri
npm install
npm run dev
```

Open the dev server URL (usually `http://localhost:1420` or `http://localhost:5173`).

### 5. (Optional) Build the Tauri Desktop App

```bash
cd sm-cache-tauri
npm run tauri build
```

---

## Usage

### Sending Queries

**Via the Dashboard:** Type a query in the Query Console at the bottom of the dashboard and press Enter or click Send.

**Via curl:**

```bash
# First query (cache miss → calls LLM)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "session_id": "demo-1"}'

# Exact repeat (Tier 1 hit, <1ms)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "session_id": "demo-1"}'

# Paraphrase (Tier 2 semantic hit)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain machine learning to me", "session_id": "demo-1"}'

# Different question (cache miss)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does photosynthesis work?", "session_id": "demo-1"}'
```

### Providing Feedback

Submit correctness feedback to calibrate conformal thresholds:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"entry_id": "<entry-id-from-response>", "was_correct": true, "similarity": 0.92}'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Cache-fronted LLM query |
| `/feedback` | POST | Correctness feedback for threshold calibration |
| `/bulk-query` | POST | Batch queries (benchmarking) |
| `/health` | GET | Health check |
| `/api/stats` | GET | Full stats for dashboard |
| `/api/analytics` | GET | Time series + distributions |
| `/api/clear-cache` | POST | Clear all cache entries |

---

## Demonstration Guide

### Demo 1: Cache Hit Behavior

1. Start the backend and frontend
2. Send: `"What is deep learning?"` — observe **MISS**, LLM response arrives
3. Send: `"What is deep learning?"` — observe **Tier 1 HIT** (exact hash, <1ms)
4. Send: `"Explain deep learning"` — observe **Tier 2 HIT** (semantic match)
5. Check the dashboard: hit rate increasing, Tier 1/Tier 2 breakdown in doughnut chart

### Demo 2: Session Context

1. Set session ID to `"context-demo"`
2. Send: `"Tell me about Tokyo"` → LLM responds about Tokyo
3. Send: `"What's the weather like there?"` → session context makes "there" = Tokyo
4. New session ID `"context-demo-2"`: `"What's the weather like there?"` → different response (no context)

### Demo 3: Admission Policy

1. Send 10+ unique queries to fill the cache
2. Watch the Admission card: total checks, admitted, rejected counts
3. One-hit-wonder queries get rejected by the frequency gate

### Demo 4: Eviction Under Pressure

1. Set `CC_CACHE_CAPACITY=10` in `.env` and restart
2. Send 15+ unique queries
3. Watch the Eviction card: evictions start triggering
4. Expensive-to-regenerate entries survive; cheap ones are evicted

---

## Analytics Dashboard

The Tauri desktop app provides a premium dark-mode analytics dashboard with:

### KPI Cards (Top Row)
- **Hit Rate** — overall cache hit percentage
- **Avg Latency** — mean query response time with P50
- **Cache Size** — current entries vs capacity
- **LLM Calls Saved** — percentage of LLM API calls avoided
- **Precision** — correctness of cache hits
- **Tier 1 Rate** — fraction of hits from exact hash lookup

### Charts
- **Hit Rate Over Time** — line chart showing convergence
- **Query Breakdown** — doughnut: Tier 1 / Tier 2 / Miss
- **Query Latency** — line chart showing latency per query
- **Thompson Sampling Arms** — bar chart of bandit arm expected rewards
- **Similarity Distribution** — histogram of hit similarities
- **Conformal Threshold Trace** — threshold evolution over time

### System Detail Panels
- Eviction (CostGDSF stats)
- Admission (W-TinyLFU stats)
- Bandit (Thompson Sampling)
- Conformal Thresholds (calibration status)
- Embedding Service (cache hits, encode time)
- LLM Provider (backend, calls, circuit state)

### Interactive Query Console
- Type queries directly in the dashboard
- See hit/miss status, tier, latency, similarity
- Session ID support for multi-turn testing

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_data_structures.py -v

# With coverage
python -m pytest tests/ -v --tb=short
```

---

## Project Structure

```
sem-cache-c2/
├── contextual_cache/          # Python backend
│   ├── __init__.py
│   ├── main.py               # FastAPI entry point
│   ├── config.py             # Pydantic settings
│   ├── models.py             # Data models
│   ├── cache_manager.py      # Central orchestrator
│   ├── lookup_engine.py      # Two-tier lookup (hash + FAISS)
│   ├── embedding_service.py  # Sentence-transformers wrapper
│   ├── session_context.py    # EMA session accumulator
│   ├── conformal_thresholds.py  # Per-entry conformal τ
│   ├── admission_policy.py   # Semantic-W-TinyLFU
│   ├── eviction.py           # CostGDSF eviction
│   ├── bandit.py             # Thompson Sampling
│   ├── drift_detection.py    # ADWIN drift detector
│   ├── circuit_breaker.py    # Per-dependency fault isolation
│   ├── llm_provider.py       # Ollama/Groq/OpenAI abstraction
│   ├── metrics.py            # Metrics collector
│   ├── data_structures.py    # CMS, LSH, LRU, SLRU
│   └── api/
│       ├── __init__.py
│       └── routes.py         # FastAPI routes
├── sm-cache-tauri/            # Tauri + SvelteKit frontend
│   ├── src/
│   │   ├── app.html
│   │   ├── lib/
│   │   │   └── api.ts        # TypeScript API client
│   │   └── routes/
│   │       ├── +layout.ts
│   │       └── +page.svelte  # Analytics dashboard
│   ├── src-tauri/             # Rust Tauri backend
│   └── package.json
├── tests/                     # Unit tests
│   ├── test_data_structures.py
│   ├── test_session_context.py
│   ├── test_conformal_thresholds.py
│   ├── test_eviction_admission.py
│   └── test_fault_tolerance.py
├── pyproject.toml
├── IMPLEMENTATION_GUIDE.md    # Architecture spec
└── README.md                 # This file
```

---

## Configuration Reference

All settings use the `CC_` prefix as environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CC_CACHE_CAPACITY` | 50000 | Max cache entries |
| `CC_TARGET_ERROR_RATE` | 0.05 | Conformal ε (P(error) ≤ ε) |
| `CC_CONTEXT_ALPHA` | 0.85 | Session fusion weight (0=context only, 1=query only) |
| `CC_EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence-transformers model |
| `CC_EMBEDDING_DIM` | 384 | Embedding dimensionality |
| `CC_LLM_BACKEND` | ollama | LLM backend (ollama/groq/openai) |
| `CC_LLM_MODEL` | llama3.2 | LLM model name |
| `CC_LLM_BASE_URL` | http://localhost:11434 | LLM API base URL |
| `CC_LLM_API_KEY` | None | API key (for Groq/OpenAI) |
| `CC_SESSION_TIMEOUT_S` | 1800 | Session expiry (30 min) |
| `CC_DEFAULT_THRESHOLD` | 0.80 | Default conformal τ for new entries |
| `CC_WINDOW_PCT` | 0.01 | W-TinyLFU window fraction |
| `CC_CB_FAILURE_THRESHOLD` | 5 | Circuit breaker failure count |
| `CC_CB_RESET_TIMEOUT_S` | 30.0 | Circuit breaker reset delay |

---

## Research References

1. vCache (Schroeder et al., 2025) — per-entry conformal thresholds
2. MeanCache (Gill et al., 2024) — federated semantic caching
3. W-TinyLFU (Einziger et al., 2017) — frequency-based admission
4. RAGCache (Jin et al., 2024) — cost-aware KV caching
5. ContextCache (Yan et al., 2025) — multi-turn context awareness
6. GDSF (Young, 2000) — cost-aware eviction
7. FedAvg (McMahan et al., 2017) — distributed parameter averaging
