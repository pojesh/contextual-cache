# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ContextualCache is a fault-tolerant adaptive semantic cache for LLM serving. It sits between applications and LLM providers (Ollama, Groq, OpenAI), serving cached responses for semantically similar queries. The system has two parts: a Python/FastAPI backend and a Tauri/SvelteKit analytics dashboard.

Key differentiators over existing systems (GPTCache, MeanCache, vCache):
- **Per-entry conformal thresholds** — each cached entry learns its own similarity threshold τ_i with formal guarantee P(false positive) ≤ ε
- **Thompson Sampling bandit** — learns the optimal default threshold for new entries via multi-armed bandit with Beta posteriors
- **ADWIN drift detection** — detects distribution shifts and resets bandit posteriors
- **Semantic-W-TinyLFU admission** — frequency-gated admission using LSH-bucketed Count-Min Sketch
- **CostGDSF eviction** — evicts cheapest-to-recompute entries first
- **Two-tier lookup** — O(1) exact-hash + O(log n) FAISS HNSW semantic search

## Commands

### Python Backend
```bash
# Activate virtualenv (Windows)
venv\Scripts\activate

# Start backend server (serves on http://localhost:8000)
python -m contextual_cache.main

# Run all tests
venv/Scripts/python -m pytest tests/ -v

# Run a single test file
venv/Scripts/python -m pytest tests/test_data_structures.py -v

# Run a specific test
venv/Scripts/python -m pytest tests/test_data_structures.py::TestCountMinSketch::test_basic_increment -v
```

### Tauri Frontend (sm-cache-tauri/)
```bash
cd sm-cache-tauri
npm install
npm run dev          # Dev server (http://localhost:1420 or :5173)
npm run build        # Production build
npm run check        # Svelte type checking
npm run tauri build   # Build desktop app (requires Rust)
```

## Architecture

### Backend (`contextual_cache/`)

`ContextualCacheManager` (cache_manager.py) is the central orchestrator. All cache operations flow through it. The query pipeline is:

1. **Tier 1 exact-hash lookup** — O(1), no embedding needed
2. **Tier 2 FAISS HNSW semantic search** — requires embedding, O(log n)
3. **On miss** — call LLM provider → admission check → store (with eviction if at capacity)

Key components wired together by the manager:
- **config.py** — Pydantic `Settings` singleton with validators. All tunables use `CC_` env prefix, loadable via `.env`
- **lookup_engine.py** — Two-tier lookup: exact hash dict + FAISS IndexIDMap2(IndexHNSWFlat). Auto-rebuilds index after N removals
- **embedding_service.py** — Sentence-transformers wrapper with LRU cache. Default model: `paraphrase-MiniLM-L6-v2`
- **session_context.py** — Multi-turn EMA fusion for session-aware queries (configurable α and decay)
- **conformal_thresholds.py** — Per-entry similarity threshold τ via online conformal prediction with sliding window
- **admission_policy.py** — Semantic-W-TinyLFU: window (5% capacity) admits freely, main cache is frequency-gated via LSH-bucketed CMS
- **eviction.py** — CostGDSF: priority = freq × regen_cost / storage_bytes + L
- **bandit.py** — Thompson Sampling over 10 threshold arms [0.65..0.95] with Beta posteriors; FedAvg sync for distributed mode
- **drift_detection.py** — ADWIN drift detector using Hoeffding bounds
- **circuit_breaker.py** — Per-dependency fault isolation (CLOSED → OPEN → HALF_OPEN) using monotonic time
- **llm_provider.py** — Unified abstraction over Ollama/Groq/OpenAI with exponential backoff, Retry-After parsing, structured timeouts. Tracks input/output token counts
- **data_structures.py** — Count-Min Sketch (conservative update + halving), RandomProjectionLSH, LRU, SLRU
- **persistence.py** — SQLite persistence for cache entries, conformal scores, bandit state, and conversation history
- **metrics.py** — Metrics collector powering the dashboard (time series, latency percentiles, similarity/threshold distributions)
- **rate_limiter.py** — Token bucket rate limiter with per-tenant support
- **utils.py** — Shared `normalize_text()` used consistently across lookup engine and embedding service
- **api/routes.py** — FastAPI route handlers

### Distributed Systems Components (optional, disabled by default)
- **consistent_hash.py** — SHA-256 consistent hashing ring with virtual nodes for multi-node sharding
- **wal.py** — Write-ahead log with CRC32 integrity checking for crash recovery
- **vector_clock.py** — Logical timestamps for causal ordering in distributed mode
- **gossip.py** — Push-pull gossip protocol for bandit posterior sync across nodes

### Benchmark System (`contextual_cache/benchmark/`)
- **runner.py** — Runs all cache strategies on NQ dataset with shared embeddings and LLM responses. Includes feedback loop for ContextualCache wrapper (conformal + bandit updates using gold answers)
- **baselines.py** — 6 baseline strategies: ExactMatchLRU, GPTCache, MeanCache, vCache, RAGCache, NoAdmission
- **dataset.py** — NQ dataset loading with deterministic paraphrase generation (~30% variants)

### Frontend (`sm-cache-tauri/`)

SvelteKit + Chart.js dashboard with dark theme. Four pages:
- `routes/+page.svelte` — Chat interface with session management, cache hit/miss metadata, token usage display, and collapsible conversation history sidebar (persisted via backend SQLite)
- `routes/compare/+page.svelte` — Side-by-side comparison: cached vs direct LLM responses with latency/token/speedup metrics
- `routes/analytics/+page.svelte` — Real-time analytics: 6 KPIs, hit rate trend, tier breakdown, latency, Thompson Sampling arms, similarity distribution, conformal threshold trace, system component stats
- `routes/benchmark/+page.svelte` — Benchmark runner UI with live progress, strategy comparison charts and detailed metrics table
- `lib/api.ts` — TypeScript API client for all backend endpoints

### API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Main cache-fronted LLM query (returns response, hit/miss, tier, latency, tokens) |
| POST | `/feedback` | Correctness feedback for conformal threshold calibration |
| POST | `/bulk-query` | Batch queries |
| POST | `/api/query-direct` | Bypass cache, call LLM directly (for comparison demos) |
| GET | `/health` | Health check |
| GET | `/api/stats` | Full analytics for dashboard |
| GET | `/api/analytics` | Detailed analytics data |
| POST | `/api/clear-cache` | Clear all cache entries |
| GET | `/api/conversations` | List chat sessions |
| GET | `/api/conversations/{session_id}` | Get session messages |
| DELETE | `/api/conversations/{session_id}` | Delete session |
| PATCH | `/api/conversations/{session_id}` | Rename session |
| POST | `/api/benchmark/run` | Start benchmark run |
| GET | `/api/benchmark/status/{run_id}` | Poll benchmark progress |
| GET | `/api/benchmark/results` | List benchmark results |
| GET | `/api/benchmark/results/{run_id}` | Get specific result |

## Testing

Tests use `pytest` with `pytest-asyncio` (asyncio_mode = auto). 120 tests across 12 files:
- `test_data_structures.py` — CMS, LSH, LRU, SLRU
- `test_session_context.py` — Session manager and EMA fusion
- `test_conformal_thresholds.py` — Threshold calibration and clamping
- `test_eviction_admission.py` — CostGDSF eviction + W-TinyLFU admission
- `test_fault_tolerance.py` — Circuit breaker, ADWIN drift, Thompson Sampling bandit
- `test_lookup_engine.py` — Two-tier store/remove/search, FAISS ID mapping, index rebuild, normalization, TTL
- `test_persistence.py` — SQLite save/load round-trip, embedding fidelity, bandit state, conformal scores
- `test_config_validation.py` — Config validators for capacity, thresholds, dimensions
- `test_rate_limiter.py` — Token bucket behavior, per-tenant isolation
- `test_consistent_hash.py` — Ring distribution, node add/remove
- `test_vector_clock.py` — Ordering, merge, concurrent detection
- `test_wal.py` — Append/replay, checkpoint, LSN recovery

## Configuration

All settings are in `config.py` via Pydantic `BaseSettings`. Override via environment variables with `CC_` prefix or a `.env` file.

Key settings:
- `CC_LLM_BACKEND` — ollama (default), groq, openai
- `CC_LLM_MODEL` — llama3.2:3b (default)
- `CC_CACHE_CAPACITY` — 50,000 (default)
- `CC_TARGET_ERROR_RATE` — 0.05 (conformal ε)
- `CC_DEFAULT_THRESHOLD` — 0.75 (similarity threshold for new entries)
- `CC_EMBEDDING_MODEL` — paraphrase-MiniLM-L6-v2 (384d)
- `CC_ANN_K` — 10 (FAISS top-k neighbors)
- `CC_WINDOW_PCT` — 0.05 (W-TinyLFU window fraction)
- `CC_PERSISTENCE_ENABLED` — false (SQLite cache entry persistence; conversation history always persists)
- `CC_DEFAULT_TTL_S` — 3600 (entry TTL, 0 = no expiry)

## Dependencies

- Python 3.10+ with PyTorch (CUDA 12.6), FastAPI, sentence-transformers, faiss-cpu, numpy, httpx, pydantic-settings
- Node.js 18+ for the Tauri frontend (Svelte 5, SvelteKit, Chart.js)
- Ollama (default LLM backend) running locally on port 11434
