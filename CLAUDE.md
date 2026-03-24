# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ContextualCache is a fault-tolerant adaptive semantic cache for LLM serving. It sits between applications and LLM providers (Ollama, Groq, OpenAI), serving cached responses for semantically similar queries. The system has two parts: a Python/FastAPI backend and a Tauri/SvelteKit analytics dashboard.

## Commands

### Python Backend
```bash
# Activate virtualenv (Windows)
venv\Scripts\activate

# Start backend server (serves on http://localhost:8000)
python -m contextual_cache.main

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_data_structures.py -v

# Run a specific test
python -m pytest tests/test_data_structures.py::TestCountMinSketch::test_basic_increment -v
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
- **config.py** — Pydantic `Settings` singleton. All tunables use `CC_` env prefix, loadable via `.env`
- **lookup_engine.py** — Two-tier lookup: exact hash dict + FAISS HNSW index
- **embedding_service.py** — Sentence-transformers wrapper with LRU cache
- **session_context.py** — Multi-turn EMA fusion for session-aware queries
- **conformal_thresholds.py** — Per-entry similarity threshold τ with formal guarantee P(error) ≤ ε
- **admission_policy.py** — Semantic-W-TinyLFU using LSH-bucketed Count-Min Sketch
- **eviction.py** — CostGDSF: evicts cheapest-to-recompute entries first
- **bandit.py** — Thompson Sampling for adaptive threshold learning
- **drift_detection.py** — ADWIN drift detector for bandit
- **circuit_breaker.py** — Per-dependency fault isolation (embedding, LLM)
- **llm_provider.py** — Unified abstraction over Ollama/Groq/OpenAI
- **data_structures.py** — Count-Min Sketch, LSH hash, LRU, SLRU implementations
- **metrics.py** — Metrics collector powering the dashboard
- **api/routes.py** — FastAPI route handlers

### Benchmark System (`contextual_cache/benchmark/`)
- **runner.py** — Runs all cache strategies on benchmark datasets, stores results as JSON in `benchmarks/results/`
- **baselines.py** — Baseline cache implementations for comparison
- **dataset.py** — Benchmark dataset loading from `benchmarks/dataset_cache/`

### Frontend (`sm-cache-tauri/`)

SvelteKit + Chart.js dashboard that polls `/api/stats` for real-time analytics. Three pages:
- `routes/+page.svelte` — Main dashboard with KPI cards, charts, query console
- `routes/analytics/+page.svelte` — Extended analytics view
- `routes/benchmark/+page.svelte` — Benchmark results visualization
- `lib/api.ts` — TypeScript API client for backend communication

## Testing

Tests use `pytest` with `pytest-asyncio` (asyncio_mode = auto). Test files:
- `test_data_structures.py` — CMS, LSH, LRU, SLRU
- `test_session_context.py` — Session manager and EMA fusion
- `test_conformal_thresholds.py` — Threshold calibration
- `test_eviction_admission.py` — CostGDSF eviction + W-TinyLFU admission
- `test_fault_tolerance.py` — Circuit breaker behavior

## Configuration

All settings are in `config.py` via Pydantic `BaseSettings`. Override via environment variables with `CC_` prefix or a `.env` file. Key settings: `CC_LLM_BACKEND`, `CC_LLM_MODEL`, `CC_CACHE_CAPACITY`, `CC_TARGET_ERROR_RATE`, `CC_EMBEDDING_MODEL`.

## Dependencies

- Python 3.10+ with PyTorch (CUDA 12.6), FastAPI, sentence-transformers, faiss-cpu, numpy, httpx
- Node.js 18+ for the Tauri frontend (Svelte 5, SvelteKit, Chart.js)
- Ollama (default LLM backend) running locally on port 11434
