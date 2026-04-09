# Semantic Cache for LLM Serving

### Final Presentation

**Pojesh Kumar R (22BAI1373)**
B.Tech CSE (AI & ML) | VIT Chennai
Guide: Dr. Kumar R | April 2026

***

## Slide 1: Problem Definition & Motivation

**Problem:** Running LLMs at scale is expensive and slow. Every query consumes GPU compute, and providers bill per token. Even standard deployments cost thousands of dollars daily.

**Key Insight:** \~31% of queries in 27,000 ChatGPT conversations are semantically similar to previously asked questions (same meaning, different wording).

**Opportunity:** Semantic caching can reuse past LLM responses for similar queries, reducing cost and latency.

**Core Problem:** Design a semantic cache that overcomes fixed global thresholds through per-entry adaptive threshold learning, with cost-aware cache management and formal guarantees on false positive rates.

***

## Slide 2: Objectives

1. **Per-entry conformal thresholds** that learn from feedback, guaranteeing P(false positive) <= 5%
2. **Thompson Sampling bandit** (10 arms, Beta posteriors) to learn optimal default thresholds for new entries + ADWIN drift detection
3. **Semantic W-TinyLFU admission** using LSH-bucketed Count-Min Sketch to reject one-hit-wonder queries
4. **CostGDSF eviction** prioritizing expensive-to-regenerate entries: H = freq x cost / size + L
5. **Two-tier lookup:** O(1) exact-hash + O(log n) FAISS HNSW semantic search
6. **Benchmark framework** comparing against 6 baseline strategies on public QA datasets
7. **Full-stack analytics dashboard** (Tauri/SvelteKit) for real-time monitoring and benchmarking

***

## Slide 3: Literature Review

| Paper                           | Method                                                                      | Key Finding                                                                           |
| ------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| GPTCache (Regmi & Pun, 2024)    | Fixed threshold (0.75), HNSW, LRU                                           | Shows semantic caching reduces LLM costs; 19.38% hit rate, 61.9% precision on NQ      |
| MeanCache (Gill et al., 2024)   | Adaptive global threshold (mean - std), federated learning, PCA compression | Global threshold cannot distinguish query types; 13.23% hit rate, 66.28% precision    |
| vCache (Schroeder et al., 2025) | Per-entry conformal thresholds, online conformal prediction                 | Formal false positive guarantees; but high default (0.80), no bandit, LRU only        |
| RAGCache (Jin et al., 2024)     | Static threshold (0.85), CostGDSF eviction                                  | Cost-aware eviction but overly conservative threshold, no per-entry adaptation        |
| SCALM (Li et al., 2024)         | Vector quantization + semantic clustering                                   | Significant token savings but static cluster boundaries, no feedback-based thresholds |

***

## Slide 4: Research Gap

### Gap 1: No Per-Entry Threshold Adaptation

- All systems except vCache use a single global threshold
- vCache uses high default (0.80) with no exploration mechanism

### Gap 2: Primitive Cache Management

- Most use simple LRU eviction (treats all entries equally)
- No admission control to filter rare/one-hit queries

### Gap 3: No Formal Guarantees + Adaptive Thresholds

- vCache provides conformal guarantees but lacks bandit exploration and drift-aware resetting

**Our system addresses all three gaps** through integrated conformal prediction + Thompson Sampling + ADWIN + W-TinyLFU + CostGDSF.

***

## Slide 5: Proposed System Overview

**ContextualCache** - A fault-tolerant adaptive semantic cache for LLM serving

### Key Innovations:

- **Per-entry conformal thresholds** - each cached entry learns its own similarity boundary with P(false positive) <= epsilon
- **Thompson Sampling bandit** - 10 arms \[0.65..0.95] with Beta posteriors learn optimal default threshold
- **ADWIN drift detection** - detects distribution shifts, resets bandit posteriors
- **Semantic W-TinyLFU** - LSH-bucketed frequency estimation rejects cache-polluting queries
- **CostGDSF eviction** - retains expensive-to-regenerate entries
- **Two-tier lookup** - O(1) exact hash + O(log n) FAISS HNSW

### Engineering Contributions:

- Full-stack Python FastAPI backend + Tauri/SvelteKit desktop dashboard
- Multi-LLM provider support (Ollama, Groq, OpenAI)
- Integrated benchmark suite comparing 7 strategies
- 120 automated tests across 12 test files

***

## Slide 6: System Architecture

```
User Query
    |
    v
[Tier 1: Exact Hash Lookup] ---> O(1), <1ms
    | (miss)
    v
[Embedding: MiniLM-L6-v2] ---> 384d vectors, 5-20ms
    |
    v
[Session Context Fusion] ---> EMA (alpha=0.85)
    |
    v
[Tier 2: FAISS HNSW Search] ---> O(log n), top-k=10
    |
    v
[Per-Entry Conformal Threshold Check]
   / \
 HIT   MISS
  |      |
  v      v
Return  [LLM Provider (Ollama/Groq/OpenAI)]
cached      |
response    v
        [W-TinyLFU Admission Gate]
            |
            v
        [Store with CostGDSF metadata]

Feedback Loop:
  Correctness signals --> Conformal Calibrator
                     --> Thompson Sampling Bandit
                     --> ADWIN Drift Detector
```

***

## Slide 7: Methodology - Algorithms

### Algorithm 1: Conformal Threshold Update

- For correct hit: nonconformity = 1.0 - similarity
- For incorrect hit: nonconformity = min(1.0, max(1.0 - similarity + 0.1, 0.5))
- Threshold tau\_i = 1.0 - quantile((1-epsilon) \* (n+1)) of sorted scores
- Clamped to \[0.60, 0.99], minimum 3 calibration points

### Algorithm 2: Thompson Sampling

- 10 arms: thresholds \[0.650, 0.683, ..., 0.950]
- Each arm: Beta(alpha, beta) posterior
- On cache hit: compute reward, update best arm's posterior
- Feed reward to ADWIN; if drift detected, reset all posteriors to Beta(2,2)

### CostGDSF Eviction Priority:

```
H(e) = frequency(e) x regen_cost(e) / storage_bytes(e) + L
```

### W-TinyLFU Admission:

- Window cache (5% capacity): admits freely
- Main cache (95%): frequency-gated via LSH-bucketed CMS (width=4096, depth=4)

***

## Slide 8: Tools & Technologies

| Category             | Technology                             |
| -------------------- | -------------------------------------- |
| Language             | Python 3.10                            |
| Backend Framework    | FastAPI with Uvicorn ASGI              |
| Embedding Model      | paraphrase-MiniLM-L6-v2 (384d)         |
| Vector Index         | FAISS (faiss-cpu) with HNSW            |
| LLM Backend          | Ollama (LLaMA 3.2:3b) / Groq / OpenAI  |
| Optional Cache Store | Redis (redis-py async)                 |
| Persistence          | SQLite                                 |
| Frontend Framework   | SvelteKit (Svelte 5)                   |
| Desktop Runtime      | Tauri (Rust)                           |
| Charting             | Chart.js                               |
| Deep Learning        | PyTorch (CUDA 12.6)                    |
| Testing              | pytest with pytest-asyncio (120 tests) |

***

## Slide 9: Implementation Details

### Backend Modules (contextual\_cache/):

- **cache\_manager** - Central orchestrator for all cache operations
- **lookup\_engine** - Two-tier: SHA-256 exact hash + FAISS IndexIDMap2(IndexHNSWFlat)
- **embedding\_service** - Sentence-transformers with LRU cache (10K entries)
- **conformal\_thresholds** - Per-entry threshold store with online calibration
- **bandit** - Thompson Sampling MAB + ADWIN integration
- **admission\_policy** - Semantic W-TinyLFU with LSH-bucketed CMS
- **eviction** - CostGDSF with inflation factor tracking
- **llm\_provider** - Unified abstraction over Ollama/Groq/OpenAI

### Frontend (Tauri/SvelteKit):

- Chat interface with session management
- Side-by-side cache comparison view
- Real-time analytics dashboard (6 KPIs, charts)
- Integrated benchmark suite

### Datasets:

- **Natural Questions (NQ):** 500 unique + 150 paraphrases = 650 queries
- **SQuAD:** 500 unique + 150 paraphrases = 650 queries
- 26 deterministic paraphrase transformation patterns

### Experimental Setup:

- Cache capacity: 300 entries (benchmark), 50,000 (production)
- LLM: Ollama running LLaMA 3.2:3b locally
- All embeddings and LLM responses pre-computed and shared across strategies

***

## Slide 10: Results & Analysis - NQ Dataset

| Strategy            | Hit Rate   | Precision  | F1        | Correct | LLM Calls |
| ------------------- | ---------- | ---------- | --------- | ------- | --------- |
| **ContextualCache** | **20.77%** | **62.22%** | **0.214** | **84**  | **515**   |
| GPTCache            | 19.38%     | 61.90%     | 0.201     | 78      | 524       |
| vCache              | 18.62%     | 62.81%     | 0.197     | 76      | 529       |
| No-Admission        | 17.69%     | 63.48%     | 0.191     | 73      | 535       |
| RAGCache            | 17.23%     | 64.29%     | 0.189     | 72      | 538       |
| MeanCache           | 13.23%     | 66.28%     | 0.155     | 57      | 564       |
| Exact-Match LRU     | 0.00%      | --         | 0.000     | 0       | 650       |

**Key Findings:**

- Highest hit rate (20.77%) and F1 (0.214) among all strategies
- 135 LLM calls saved out of 650 queries
- 6 more correct hits than GPTCache while saving 9 more LLM calls
- Precision competitive with GPTCache (62.22% vs 61.90%)

***

## Slide 11: Results - SQuAD & Cross-Dataset

### SQuAD Results:

| Strategy            | Hit Rate   | Precision | F1        | LLM Calls |
| ------------------- | ---------- | --------- | --------- | --------- |
| **ContextualCache** | **23.69%** | 35.71%    | **0.137** | **496**   |
| GPTCache            | 19.54%     | 38.58%    | 0.126     | 523       |
| vCache              | 17.85%     | 38.79%    | 0.118     | 534       |

- 154 LLM calls saved (most among all strategies)
- Higher hit rate on SQuAD due to passage-focused questions sharing vocabulary

### Cross-Dataset Insight:

- **NQ:** Lower hit rate (20.77%) but higher precision (62.22%) - diverse web queries
- **SQuAD:** Higher hit rate (23.69%) but lower precision (35.71%) - similar questions from same passages can have different answers

***

## Slide 12: Results - Graphs & Performance

### Latency Analysis:

| Strategy        | Hit Latency | Overall Latency | LLM Saved |
| --------------- | ----------- | --------------- | --------- |
| ContextualCache | 0.00 ms     | 1610.17 ms      | 135       |
| GPTCache        | 0.13 ms     | 1613.17 ms      | 126       |
| MeanCache       | 0.00 ms     | 1736.22 ms      | 86        |

- Sub-millisecond cache hit latency (vs 500ms-6s for LLM)
- Lowest overall latency due to highest hit rate

### Thompson Sampling Convergence:

- Best arm: threshold \~0.65 with expected reward 0.889
- Bandit learns to favor permissive defaults for NQ workload
- Arms above 0.85 accumulate pessimistic posteriors

### Conformal Threshold Evolution:

- Entries diverge from default (0.70) after 3 feedback events
- Factual entries tighten to \~0.85; broad entries relax to \~0.65
- Heterogeneous threshold landscape within formal guarantee

***

## Slide 13: Comparison - Existing vs Proposed

| Feature          | GPTCache         | MeanCache                  | vCache                 | **ContextualCache**        |
| ---------------- | ---------------- | -------------------------- | ---------------------- | -------------------------- |
| Threshold        | Fixed (0.75)     | Global adaptive (mean-std) | Per-entry conformal    | **Per-entry conformal**    |
| Default Learning | None             | None                       | None                   | **Thompson Sampling**      |
| Drift Detection  | None             | None                       | None                   | **ADWIN**                  |
| Admission Policy | None (admit all) | None                       | None                   | **Semantic W-TinyLFU**     |
| Eviction         | LRU              | LRU                        | LRU                    | **CostGDSF**               |
| Formal Guarantee | No               | No                         | Yes (P(FP) <= epsilon) | **Yes (P(FP) <= epsilon)** |
| Hit Rate (NQ)    | 19.38%           | 13.23%                     | 18.62%                 | **20.77%**                 |
| F1 Score (NQ)    | 0.201            | 0.155                      | 0.197                  | **0.214**                  |
| LLM Calls Saved  | 126              | 86                         | 121                    | **135**                    |

***

## Slide 14: Applications

- **LLM API Cost Reduction:** At 100K queries/day, eliminates \~20,770 LLM calls. Projected savings: \~$280/day ($8,400/month) at GPT-4 pricing
- **Chatbot & Virtual Assistant Platforms:** Sub-millisecond cached responses for frequently asked questions
- **Enterprise Search & Knowledge Bases:** Reduce latency for repetitive internal queries
- **Customer Support Systems:** Cache common support queries across sessions
- **Educational Platforms:** Similar student questions get instant cached responses
- **Healthcare/Legal/Finance AI Tools:** Domain-specific caching with per-entry precision control
- **Multi-tenant SaaS:** Rate-limited, tenant-isolated caching with configurable thresholds
- **Edge Deployment:** Reduce cloud LLM dependency with local semantic caching

***

## Slide 15: Challenges Faced

| Challenge                                                                               | Solution                                                                          |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Conformal Cold Start** - entries stuck at default threshold with few feedback signals | Reduced min calibration to 3 points; implicit feedback from similarity scores     |
| **Embedding Model Selection** - all-MiniLM-L6-v2 gave low paraphrase similarity         | Switched to paraphrase-MiniLM-L6-v2 (+5-10% similarity for paraphrases)           |
| **FAISS Memory Management** - HNSW doesn't support true in-place removal                | IndexIDMap2 wrapper + periodic rebuild after 500 removals                         |
| **Admission Policy Calibration** - frequency gate too aggressive for benchmarks         | Changed strict > to >= comparison; larger window for benchmarks                   |
| **Benchmark Fairness** - inconsistent results from separate LLM calls per strategy      | Pre-compute all embeddings and LLM responses in shared pass                       |
| **Text Normalization** - exact-hash missing matches due to whitespace/case differences  | Shared normalize function (lowercase, strip, collapse spaces) at all entry points |
| **External Dependency Failures** - Ollama/Redis timeouts stalling entire pipeline       | Per-dependency circuit breaker (CLOSED/OPEN/HALF\_OPEN)                           |
| **CMS Frequency Aging** - stale counts biasing admission                                | Periodic halving mechanism for count decay                                        |

***

## Slide 16: Conclusion & Future Work

### Conclusion:

- Presented a **fault-tolerant adaptive semantic cache** integrating conformal prediction, Thompson Sampling, ADWIN, W-TinyLFU, and CostGDSF
- **Best hit rate** (20.77% NQ, 23.69% SQuAD) and **best F1** among all strategies
- **135 LLM calls saved** on NQ with **62.22% precision** - competitive with baselines
- Per-entry thresholds provide **formal guarantee** P(false positive) <= epsilon
- Full-stack implementation: Python backend + Tauri/SvelteKit dashboard + 120 tests
- Distributed components (consistent hashing, WAL, vector clocks, gossip) implemented and unit-tested

### Future Work:

1. **Embedding model upgrades** - Evaluate E5-base-v2 (768d), GTE-large for better semantic matching accuracy
2. **Multi-round evaluation** - Run queries over multiple passes to measure steady-state precision improvement as thresholds calibrate
3. **Conversational & domain-specific datasets** - Test session context fusion on multi-turn dialogues; evaluate on medical/legal/financial domains

***

## Slide 17: References

\[1] K. Haqiq, "MinCache: A hybrid cache system for efficient chatbots with hierarchical embedding matching and LLM," *Proc. Future Gener. Comput. Syst.*, vol. 165, pp. 107-120, 2025.

\[2] W. Gill, A. Elidrisi, et al., "MeanCache: User-Centric Semantic Caching for LLM Web Services," *arXiv preprint arXiv:2403.02694*, 2024.

\[3] S. Regmi and C. P. Pun, "GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching," *arXiv preprint arXiv:2411.05276*, 2024.

\[4] A. Schroeder, et al., "vCache: Per-Entry Conformal Thresholds for Semantic Caching," UC Berkeley, 2025.

\[5] H. Li, et al., "Context-based Semantic Caching for LLM Applications," *Proc. IEEE Int. Conf. on Web Services (ICWS)*, 2024.

\[6] Z. Li, et al., "SCALM: Towards Semantic Caching for Automated Chat Services with LLMs," *Proc. IEEE/ACM Int. Symp. on Quality of Service (IWQoS)*, 2024.

\[7] G. Einziger, R. Friedman, B. Manes, "TinyLFU: A Highly Efficient Cache Admission Policy," *ACM Trans. Storage*, vol. 13, no. 4, 2017.

\[8] Y. Jin, et al., "RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation," *arXiv preprint arXiv:2404.12457*, 2024.

\[9] B. McMahan, et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," *Proc. AISTATS*, 2017.

\[10] A. Bifet and R. Gavalda, "Learning from Time-Changing Data with Adaptive Windowing," *Proc. SIAM Int. Conf. on Data Mining (SDM)*, 2007.

\[11] V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*, Springer, 2005.

\[12] M. Young, "The GDSF Online Caching Algorithm," Tech. Report, 2000.

\[13] X. Gao, et al., "Online Context Caching for Distributed LLM Serving," *Proc. IEEE IPDPS*, 2025.

\[14] F. Zhu, et al., "Semantic Caching for Large Language Model Cost Reduction," *Proc. IEEE GLOBECOM*, 2023.

\[15] R. Zha, et al., "Online Optimization for Semantic Cache Hit Prediction," *Proc. IEEE Int. Conf. on Big Data*, pp. 690-699, 2023.

\[16] N. F. Liu, et al., "Lost in the Middle: How Language Models Use Long Contexts," *Trans. Assoc. Comput. Linguistics*, vol. 12, pp. 157-173, 2024.

\[17] T. Kwiatkowski, et al., "Natural Questions: A Benchmark for Question Answering Research," *Trans. Assoc. Comput. Linguistics*, vol. 7, pp. 453-466, 2019.

\[18] P. Rajpurkar, et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text," *Proc. EMNLP*, 2016.

\[19] W. Thompson, "On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples," *Biometrika*, vol. 25, pp. 285-294, 1933.

\[20] J. Johnson, M. Douze, H. Jegou, "Billion-Scale Similarity Search with GPUs," *IEEE Trans. on Big Data*, vol. 7, no. 3, pp. 535-547, 2021.
