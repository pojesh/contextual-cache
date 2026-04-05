# Poster Content — Fill into poster-template.pptx

---

## HEADER

**Project Title:** Semantic Cache with Adaptive Thresholds for LLM Serving

**Student Name:** Pojesh Kumar R (22BAI1373)
**Guide Name:** Dr. Kumar R
**School Of Computer Science And Engineering**

---

## LEFT COLUMN

### MOTIVATION / INTRODUCTION

- Large Language Models power chatbots, search engines, and content generation, but inference costs are high ($0.03–$0.06 per 1K tokens) and latency ranges from 500 ms to 6 seconds per query.

- Research on 27,000 ChatGPT conversations reveals that 31% of queries are semantically similar to previously asked questions, presenting a major caching opportunity.

- Existing semantic caches like GPTCache and MeanCache use a single global similarity threshold for all entries, leading to either excessive false positives or missed cache opportunities across different query types.

- Our system, ContextualCache, replaces the global threshold with per-entry conformal thresholds that learn individually for each cached entry, with formal guarantees on false positive rates.

### SDG GOAL NUMBER: < 9 >

### OBJECTIVES

- Design per-entry conformal similarity thresholds with formal guarantee P(false positive) ≤ 5% for each cached entry.

- Implement Thompson Sampling bandit (10 arms, Beta posteriors) to learn optimal default thresholds for new entries, with ADWIN drift detection for automatic resets.

- Develop Semantic W-TinyLFU admission policy using LSH-bucketed Count-Min Sketch to reject infrequent queries that would pollute the cache.

- Implement CostGDSF eviction prioritizing retention of entries expensive to regenerate via LLM.

- Build two-tier lookup: O(1) exact-hash + O(log n) FAISS HNSW semantic search.

- Benchmark against 6 baseline strategies on public QA datasets (Natural Questions, SQuAD).

### SCOPE OF THE PROJECT

- Backend: Python/FastAPI supporting Ollama, Groq, and OpenAI as LLM providers.

- Embedding: paraphrase-MiniLM-L6-v2 (384-dimensional dense vectors).

- Evaluation on Natural Questions and SQuAD datasets (650 queries each, 30% paraphrase variants).

- Full-stack analytics dashboard built with Tauri/SvelteKit for real-time monitoring, cache comparison, and benchmark visualization.

- Distributed components (consistent hashing, WAL, vector clocks, gossip protocol) implemented for future multi-node scaling.

- 120 automated tests across 12 test files covering all system components.

---

## MIDDLE COLUMN

### METHODOLOGY

The system uses a multi-stage query pipeline:

1. Tier 1 — Exact Hash Lookup: Query text is normalized and hashed (SHA-256). Checked against in-memory dictionary in <1 ms.

2. Tier 2 — Semantic Search: On Tier 1 miss, query is embedded using paraphrase-MiniLM-L6-v2 (384d) and searched against FAISS HNSW index (top-k=10 neighbors).

3. Conformal Threshold Check: Each candidate's cosine similarity is compared against its per-entry threshold τ_i, computed as the (1−ε) quantile of nonconformity scores.

4. LLM Fallback: On cache miss, query is sent to LLM (Ollama/Groq/OpenAI). Response passes through W-TinyLFU admission gate before storage.

5. Feedback Loop: Correctness signals update conformal thresholds, Thompson Sampling bandit posteriors, and ADWIN drift detector.

Key Algorithms:
- Per-entry conformal thresholds: τ_i = 1 − quantile(nonconformity scores), clamped to [0.60, 0.99]
- Thompson Sampling: 10 arms (0.65–0.95), Beta(α,β) posteriors, reward based on similarity
- CostGDSF eviction: H(e) = freq × regen_cost / size + L

### ARCHITECTURE

[INSERT fig-architecture.png HERE — from docs/report-assets/fig-architecture.png]

---

## RIGHT COLUMN

### RESULTS

Natural Questions Dataset (650 queries, 300 capacity):

| Strategy       | Hit Rate | Precision | F1    | LLM Calls |
|---------------|----------|-----------|-------|-----------|
| Ours          | 20.77%   | 62.22%    | 0.214 | 515       |
| GPTCache      | 19.38%   | 61.90%    | 0.201 | 524       |
| vCache        | 18.62%   | 62.81%    | 0.197 | 529       |
| RAGCache      | 17.23%   | 64.29%    | 0.189 | 538       |
| MeanCache     | 13.23%   | 66.28%    | 0.155 | 564       |

- Best hit rate (20.77%) and F1 score (0.214) among all 7 strategies
- 84 correct cache hits, saving 135 LLM inference calls
- Sub-millisecond cache hit latency (<1 ms vs 500–6000 ms for LLM)
- On SQuAD: 23.69% hit rate, 154 LLM calls saved

[INSERT fig-hit-rate-precision.png HERE — from docs/research-paper/fig-hit-rate-precision.png]

### CONCLUSION

- ContextualCache outperforms GPTCache, MeanCache, vCache, and RAGCache on both NQ and SQuAD datasets in hit rate and F1 score.

- Per-entry conformal thresholds provide formal P(false positive) ≤ ε guarantees while adapting individually to each cached entry.

- Thompson Sampling bandit learns optimal default threshold (~0.70), capturing paraphrase matches missed by conservative fixed thresholds.

- W-TinyLFU admission improves hit rate by 3.08 percentage points by filtering one-time queries.

- Projected monthly savings of $8,400+ for a service handling 100K queries/day with GPT-4 pricing.

### CONTACT DETAILS

**MAILID:** pojeshkumar.r2022@vitstudent.ac.in
**MOBILE NO:** [YOUR PHONE NUMBER]

### REFERENCES

- Regmi & Pun, "GPT Semantic Cache," arXiv:2411.05276, 2024.
- Gill et al., "MeanCache," arXiv:2403.02694, 2024.
- Schroeder et al., "vCache," UC Berkeley, 2025.
- Einziger et al., "TinyLFU," ACM Trans. Storage, 2017.
- Bifet & Gavalda, "ADWIN," SIAM SDM, 2007.
- Vovk et al., Algorithmic Learning in a Random World, Springer, 2005.
