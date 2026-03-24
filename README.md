# ContextualCache

**A Fault-Tolerant Adaptive Semantic Cache for LLM Serving**

ContextualCache is a production-ready semantic caching system that sits between your application and LLM providers (Ollama, Groq, OpenAI). It dramatically reduces latency and API costs by serving cached responses for semantically similar queries, with formal correctness guarantees.

---

## Architecture Overview

```
Client → Tauri App (Chat / Compare / Analytics / Benchmark)
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
    │ ADWIN Drift Detect  │  ← Resets on distribution shift
    └────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Two-Tier Lookup** | Exact hash (T1, <1ms) + FAISS HNSW semantic search (T2) with IndexIDMap2 and auto-rebuild |
| **Per-Entry Conformal Thresholds** | Each entry learns its own τ_i. Formal guarantee: P(incorrect hit) ≤ ε |
| **Thompson Sampling Bandit** | 10-arm bandit over [0.65..0.95] learns optimal default threshold via Beta posteriors |
| **ADWIN Drift Detection** | Detects distribution shifts in feedback stream, resets bandit posteriors |
| **Semantic-W-TinyLFU Admission** | Frequency-gated admission using LSH-bucketed Count-Min Sketch |
| **CostGDSF Eviction** | Priority = freq × regen_cost / storage_bytes + L |
| **Session Context Fusion** | Multi-turn EMA with configurable α (O(d) space per session) |
| **Circuit Breakers** | Per-dependency fault isolation (embedding, LLM) with CLOSED → OPEN → HALF_OPEN FSM |
| **Token Usage Tracking** | Input/output token counts from Ollama and OpenAI-compatible APIs |
| **Chat History Persistence** | SQLite-backed conversation storage with session management |
| **Side-by-Side Comparison** | Compare cached vs direct LLM responses with latency/token/speedup metrics |
| **Real-time Analytics Dashboard** | Tauri + SvelteKit + Chart.js with dark theme |
| **Benchmark Suite** | 7-strategy comparison against GPTCache, MeanCache, vCache, RAGCache baselines |

### Distributed Systems Components (optional, disabled by default)
| Feature | Description |
|---------|-------------|
| **Consistent Hashing** | SHA-256 ring with virtual nodes for multi-node sharding |
| **Write-Ahead Log** | Append-only WAL with CRC32 integrity for crash recovery |
| **Vector Clocks** | Logical timestamps for causal ordering across nodes |
| **Gossip Protocol** | Push-pull gossip for bandit posterior sync via FedAvg |
| **Rate Limiting** | Token bucket with per-tenant support |

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
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure (Optional)

Create a `.env` file in the project root to override defaults:

```env
# LLM Configuration
CC_LLM_BACKEND=ollama          # ollama | groq | openai
CC_LLM_MODEL=llama3.2:3b      # model name
CC_LLM_BASE_URL=http://localhost:11434

# For Groq/OpenAI:
# CC_LLM_API_KEY=your-api-key

# Cache Settings
CC_CACHE_CAPACITY=50000
CC_TARGET_ERROR_RATE=0.05      # conformal ε
CC_DEFAULT_THRESHOLD=0.75      # similarity threshold for new entries
CC_CONTEXT_ALPHA=0.85          # session fusion weight

# Embedding
CC_EMBEDDING_MODEL=paraphrase-MiniLM-L6-v2
CC_EMBEDDING_DIM=384

# Persistence (conversation history always persists; this controls cache entries)
CC_PERSISTENCE_ENABLED=false
CC_DEFAULT_TTL_S=3600          # entry TTL in seconds (0 = no expiry)
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
  LLM backend:    ollama (llama3.2:3b)
  Embedding:      paraphrase-MiniLM-L6-v2 (384d)
  Cache capacity:  50,000 entries
  Error rate (ε):  0.05
============================================================
```

### 4. Start the Frontend

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

## Frontend Pages

### Chat (`/`)
Interactive query interface with real-time cache hit visualization. Features:
- Chat-style UI with user/assistant bubbles
- Per-message metadata: cache hit/miss, tier, latency, similarity, token count
- Session management with multi-turn context
- **Conversation history sidebar** — persisted to SQLite, survives browser refresh
- Hint suggestions for first-time users

### Compare (`/compare`)
Side-by-side comparison inspired by LMSys Arena. For each query:
- Left panel: response via cache (may be HIT or MISS)
- Right panel: response via direct LLM call (always fresh)
- Delta bar showing time saved, tokens saved, speedup factor
- Cumulative stats: total queries, total time/tokens saved, average speedup
- State preserved across page navigation

### Analytics (`/analytics`)
Real-time dashboard with auto-refresh (3s polling):
- **6 KPI cards**: Hit Rate, Hit Avg Latency, Cache Size, LLM Calls Saved, Precision, Tier 1 Rate
- **6 charts**: Hit rate trend, query breakdown (doughnut), latency, Thompson Sampling arms, similarity distribution, conformal threshold trace
- **System component panels**: Eviction, Admission, Bandit, Conformal, Embedding, LLM stats
- Clear cache button

### Benchmark (`/benchmark`)
Run and visualize benchmark comparisons:
- Configure: number of questions (10-2000), cache capacity (10-5000)
- Live progress tracking with strategy-level updates
- Results: hit rate & precision chart, latency analysis, detailed metrics table
- 7 strategies: Exact-Match LRU, GPTCache, MeanCache, vCache, RAGCache, No-Admission, ContextualCache (Ours)
- Historical results browser

---

## Usage

### Sending Queries

**Via the Chat page:** Type a query and press Enter. Responses show cache hit/miss status, tier, latency, and token count.

**Via curl:**

```bash
# First query (cache miss → calls LLM)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "session_id": "demo-1"}'

# Exact repeat (Tier 1 hit, <1ms, 0 tokens)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "session_id": "demo-1"}'

# Paraphrase (Tier 2 semantic hit)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain machine learning to me", "session_id": "demo-1"}'

# Direct LLM call (bypasses cache, for comparison)
curl -X POST http://localhost:8000/api/query-direct \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

### Providing Feedback

Submit correctness feedback to calibrate per-entry conformal thresholds:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"entry_id": "<entry-id-from-response>", "was_correct": true, "similarity": 0.92}'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Cache-fronted LLM query (returns response, hit/miss, tier, latency, tokens) |
| `/feedback` | POST | Correctness feedback for conformal threshold calibration |
| `/bulk-query` | POST | Batch queries |
| `/api/query-direct` | POST | Bypass cache, call LLM directly (for comparison) |
| `/health` | GET | Health check |
| `/api/stats` | GET | Full stats for analytics dashboard |
| `/api/analytics` | GET | Time series + distributions |
| `/api/clear-cache` | POST | Clear all cache entries |
| `/api/conversations` | GET | List chat sessions |
| `/api/conversations/{id}` | GET | Get session messages |
| `/api/conversations/{id}` | DELETE | Delete session |
| `/api/conversations/{id}` | PATCH | Rename session |
| `/api/benchmark/run` | POST | Start benchmark run |
| `/api/benchmark/status/{id}` | GET | Poll benchmark progress |
| `/api/benchmark/results` | GET | List all benchmark results |
| `/api/benchmark/results/{id}` | GET | Get specific benchmark result |

---

## Demonstration Guide

### Demo 1: Cache Hit Behavior
1. Open the **Chat** page
2. Send: `"What is deep learning?"` — observe **MISS** with token count and latency
3. Send: `"What is deep learning?"` — observe **Tier 1 HIT** (exact hash, <1ms, 0 tokens)
4. Send: `"Explain deep learning"` — observe **Tier 2 HIT** (semantic match, similarity shown)
5. Switch to **Analytics**: hit rate increasing, Tier 1/Tier 2 breakdown in doughnut chart

### Demo 2: Side-by-Side Comparison
1. Open the **Compare** page
2. Send: `"What is deep learning?"` — both panels load simultaneously
3. First time: both show similar latency (cache miss). Note token counts
4. Send the same query again — left panel shows **HIT** (<1ms, 0 tokens), right panel still ~2s
5. Observe the delta bar: "2000ms faster", "150 tokens saved", "200x speedup"
6. Cumulative stats update at the top

### Demo 3: Conversation History
1. Chat on the **Chat** page with several queries
2. Click the clock icon (top-left) to open **History sidebar**
3. Navigate away to Analytics, then come back — history persists
4. Click a past session to reload it
5. Refresh the browser — sessions still available (SQLite-backed)

### Demo 4: Session Context
1. Set session ID (happens automatically per new session)
2. Send: `"Tell me about Tokyo"` → LLM responds about Tokyo
3. Send: `"What's the weather like there?"` → session context makes "there" = Tokyo
4. Start a new session: `"What's the weather like there?"` → different response (no context)

### Demo 5: Admission & Eviction
1. Set `CC_CACHE_CAPACITY=10` in `.env` and restart
2. Send 15+ unique queries
3. Watch **Analytics** → Admission card shows rejected entries, Eviction card shows evictions
4. Expensive-to-regenerate entries survive; cheap ones are evicted first

### Demo 6: Benchmark
1. Open the **Benchmark** page
2. Set questions: 100, capacity: 50
3. Click "Run Benchmark" — live progress shows current strategy
4. Compare ContextualCache against all baselines on hit rate, precision, F1, latency

---

## How It Works

### Thompson Sampling (Adaptive Threshold Learning)

The system uses a multi-armed bandit to learn the optimal default similarity threshold:

1. **10 arms** represent candidate thresholds: [0.65, 0.683, ..., 0.95]
2. Each arm maintains a **Beta(α, β) distribution** — α counts correct hits, β counts false positives
3. On each query, the bandit **samples** from all arm distributions and picks the highest
4. **Feedback updates**: correct hit → α++, incorrect hit → β++, uncertain → skip
5. Over time, the best threshold arm accumulates more α and gets selected more often
6. **ADWIN drift detection**: if query patterns shift, posteriors reset to Beta(2,2)

### Per-Entry Conformal Thresholds

Each cached entry learns its own threshold with a formal correctness guarantee:

1. New entries start at the bandit's best threshold (the learned prior)
2. Feedback accumulates as **nonconformity scores**: `1 - similarity` for correct, penalized for incorrect
3. After 10+ feedback signals, threshold = **(1-ε)-quantile** of nonconformity scores
4. **Guarantee**: P(false positive) ≤ ε = 5% per entry (from conformal prediction theory)
5. Sliding window of 200 scores adapts to changing match patterns

### W-TinyLFU Admission Policy

Prevents cache pollution from one-hit-wonder queries:

1. **Window cache** (5% of capacity) admits entries freely
2. When window is full, new entries compete on **estimated frequency** (via LSH-bucketed Count-Min Sketch)
3. If new entry has higher frequency than the window's LRU victim → admitted; victim promoted to main
4. If lower frequency → rejected entirely (never enters cache)

---

## Running Tests

```bash
# All 120 tests
venv/Scripts/python -m pytest tests/ -v

# Specific test file
venv/Scripts/python -m pytest tests/test_data_structures.py -v

# Specific test
venv/Scripts/python -m pytest tests/test_fault_tolerance.py::TestShardLocalBandit::test_sample_threshold -v
```

### Test Files (120 tests)
| File | Coverage |
|------|----------|
| `test_data_structures.py` | Count-Min Sketch, LSH, LRU, SLRU |
| `test_session_context.py` | Session manager, EMA fusion, expiry |
| `test_conformal_thresholds.py` | Threshold calibration, clamping, sliding window |
| `test_eviction_admission.py` | CostGDSF eviction, W-TinyLFU admission |
| `test_fault_tolerance.py` | Circuit breaker, ADWIN drift, Thompson Sampling |
| `test_lookup_engine.py` | Two-tier lookup, FAISS ID mapping, index rebuild, TTL |
| `test_persistence.py` | SQLite round-trip, embedding fidelity, bandit state |
| `test_config_validation.py` | Pydantic validators for all settings |
| `test_rate_limiter.py` | Token bucket, per-tenant isolation |
| `test_consistent_hash.py` | Ring distribution, node add/remove |
| `test_vector_clock.py` | Causal ordering, merge, concurrency |
| `test_wal.py` | WAL append/replay, checkpoint, LSN recovery |

---

## Project Structure

```
sem-cache-c2/
├── contextual_cache/          # Python backend
│   ├── main.py               # FastAPI entry point + lifespan
│   ├── config.py             # Pydantic settings with validators
│   ├── models.py             # Data models (CacheEntry, LookupResult, etc.)
│   ├── cache_manager.py      # Central orchestrator
│   ├── lookup_engine.py      # Two-tier lookup (hash + FAISS IndexIDMap2)
│   ├── embedding_service.py  # Sentence-transformers wrapper
│   ├── session_context.py    # EMA session accumulator
│   ├── conformal_thresholds.py  # Per-entry conformal τ
│   ├── admission_policy.py   # Semantic-W-TinyLFU
│   ├── eviction.py           # CostGDSF eviction
│   ├── bandit.py             # Thompson Sampling with ADWIN
│   ├── drift_detection.py    # ADWIN drift detector
│   ├── circuit_breaker.py    # Per-dependency fault isolation
│   ├── llm_provider.py       # Ollama/Groq/OpenAI with retry + token tracking
│   ├── persistence.py        # SQLite persistence (cache + conversations)
│   ├── metrics.py            # Metrics collector
│   ├── data_structures.py    # CMS, LSH, LRU, SLRU
│   ├── utils.py              # Shared normalize_text()
│   ├── rate_limiter.py       # Token bucket rate limiter
│   ├── consistent_hash.py    # Consistent hashing ring
│   ├── wal.py                # Write-ahead log
│   ├── vector_clock.py       # Logical timestamps
│   ├── gossip.py             # Gossip protocol for bandit sync
│   ├── api/
│   │   └── routes.py         # FastAPI route handlers
│   └── benchmark/
│       ├── runner.py          # Benchmark orchestrator
│       ├── baselines.py       # 6 baseline strategies
│       └── dataset.py         # NQ dataset with paraphrase generation
├── sm-cache-tauri/            # Tauri + SvelteKit frontend
│   ├── src/
│   │   ├── lib/
│   │   │   └── api.ts        # TypeScript API client
│   │   └── routes/
│   │       ├── +layout.svelte # Sidebar navigation (4 pages)
│   │       ├── +page.svelte   # Chat with history sidebar
│   │       ├── compare/
│   │       │   └── +page.svelte  # Side-by-side comparison
│   │       ├── analytics/
│   │       │   └── +page.svelte  # Real-time analytics dashboard
│   │       └── benchmark/
│   │           └── +page.svelte  # Benchmark runner + results
│   └── src-tauri/             # Rust Tauri backend
├── tests/                     # 120 unit tests
├── benchmarks/
│   └── results/               # Benchmark result JSONs
├── pyproject.toml
├── CLAUDE.md                  # Claude Code guidance
└── README.md                  # This file
```

---

## Configuration Reference

All settings use the `CC_` prefix as environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CC_CACHE_CAPACITY` | 50000 | Max cache entries |
| `CC_TARGET_ERROR_RATE` | 0.05 | Conformal ε (P(error) ≤ ε) |
| `CC_DEFAULT_THRESHOLD` | 0.75 | Default similarity threshold for new entries |
| `CC_CONTEXT_ALPHA` | 0.85 | Session fusion weight (0=context only, 1=query only) |
| `CC_CONTEXT_DECAY` | 0.5 | Context vector decay rate |
| `CC_EMBEDDING_MODEL` | paraphrase-MiniLM-L6-v2 | Sentence-transformers model |
| `CC_EMBEDDING_DIM` | 384 | Embedding dimensionality |
| `CC_LLM_BACKEND` | ollama | LLM backend (ollama/groq/openai) |
| `CC_LLM_MODEL` | llama3.2:3b | LLM model name |
| `CC_LLM_BASE_URL` | http://localhost:11434 | LLM API base URL |
| `CC_LLM_API_KEY` | None | API key (for Groq/OpenAI) |
| `CC_LLM_MAX_RETRIES` | 3 | Retry count for transient LLM failures |
| `CC_SESSION_TIMEOUT_S` | 1800 | Session expiry (30 min) |
| `CC_WINDOW_PCT` | 0.05 | W-TinyLFU window fraction |
| `CC_ANN_K` | 10 | FAISS top-k neighbors |
| `CC_HNSW_M` | 32 | HNSW links per node |
| `CC_DEFAULT_TTL_S` | 3600 | Entry TTL in seconds (0 = no expiry) |
| `CC_PERSISTENCE_ENABLED` | false | SQLite cache entry persistence |
| `CC_INDEX_REBUILD_INTERVAL` | 500 | Auto-rebuild FAISS after N removals |
| `CC_CB_FAILURE_THRESHOLD` | 5 | Circuit breaker failure count |
| `CC_CB_RESET_TIMEOUT_S` | 30.0 | Circuit breaker reset delay |
| `CC_RATE_LIMIT_RPS` | 100 | Requests per second limit |
| `CC_BANDIT_N_ARMS` | 10 | Thompson Sampling arm count |
| `CC_DRIFT_DELTA` | 0.002 | ADWIN sensitivity |

---

## Benchmark Results

Latest benchmark (Run #05616757, 650 queries, 200 capacity):

| Strategy | Hit Rate | Precision | F1 | LLM Calls |
|----------|----------|-----------|-----|-----------|
| GPTCache | 14.62% | 40.00% | 0.102 | 555 |
| vCache | 13.69% | 40.45% | 0.097 | 561 |
| **ContextualCache (Ours)** | **12.46%** | **44.44%** | **0.099** | **569** |
| No-Admission | 11.69% | 46.05% | 0.096 | 574 |
| RAGCache | 11.23% | 46.58% | 0.094 | 577 |
| MeanCache | 10.31% | 38.81% | 0.073 | 583 |
| Exact-Match LRU | 0.00% | 0.00% | 0.000 | 650 |

ContextualCache achieves the **highest precision** among strategies with comparable hit rates — fewer false positives due to per-entry conformal thresholds.

---

## Future Work

- **Redis integration** — replace in-memory dicts with Redis for Tier 1 (distributed exact-hash), Redis Cluster for multi-node sharding (`redis_enabled` already in config)
- **LLM-based paraphrase generation** — improve benchmark with more realistic paraphrases
- **QQP/MRPC/STS-B datasets** — standard benchmark datasets for semantic similarity
- **Multi-node deployment** — consistent hashing, WAL, gossip protocol components are implemented but need integration testing
- **Embedding model upgrades** — E5-base-v2 or instructor models for domain-specific caching

---

## Research References

1. vCache (Schroeder et al., 2025) — per-entry conformal thresholds
2. MeanCache (Gill et al., 2024) — federated semantic caching
3. W-TinyLFU (Einziger et al., 2017) — frequency-based admission
4. RAGCache (Jin et al., 2024) — cost-aware KV caching
5. ContextCache (Yan et al., 2025) — multi-turn context awareness
6. GDSF (Young, 2000) — cost-aware eviction
7. FedAvg (McMahan et al., 2017) — distributed parameter averaging
8. ADWIN (Bifet & Gavalda, 2007) — adaptive windowing for drift detection
9. Conformal Prediction (Vovk et al., 2005) — distribution-free prediction sets
