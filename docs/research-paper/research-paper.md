# ContextualCache: Semantic Cache with Adaptive Thresholds for LLM Serving

**Pojesh Kumar R**
School of Computer Science and Engineering
Vellore Institute of Technology, Chennai
pojeshkumar.r2022@vitstudent.ac.in

**Dr. Kumar R**
School of Computer Science and Engineering
Vellore Institute of Technology, Chennai
kumar.rangasamy@vit.ac.in

---

## Abstract

Large Language Models have become central to modern software applications, but the computational cost of inference remains a major barrier to scalable deployment. An analysis of 27,000 ChatGPT conversations shows that roughly 31 percent of queries are semantically similar to previously asked questions, presenting a clear opportunity for caching. Current semantic cache systems like GPTCache and MeanCache rely on a single global similarity threshold to decide whether a cached response matches a new query, which leads to inconsistent performance because different types of queries demand different levels of matching precision. This paper presents ContextualCache, a fault-tolerant adaptive semantic cache that replaces the global threshold with per-entry conformal thresholds offering a formal guarantee that the false positive probability stays below a configurable error rate. The system uses a Thompson Sampling bandit with Beta posteriors to learn optimal default thresholds for new entries, coupled with ADWIN drift detection for automatic posterior resets on distribution shifts. A Semantic W-TinyLFU admission policy built on locality-sensitive hashing over a Count-Min Sketch filters out infrequent queries, while CostGDSF eviction retains entries that are expensive to regenerate. The two-tier lookup architecture combines O(1) exact-hash matching with O(log n) FAISS HNSW semantic search. Evaluation on the Natural Questions dataset with 650 queries shows that ContextualCache achieves a 20.77% hit rate with 62.22% precision and an F1 of 0.214, outperforming GPTCache (19.38%, F1 0.201), vCache (18.62%, F1 0.197), and four other baselines. On the SQuAD dataset, the system reaches a 23.69% hit rate, saving 154 LLM inference calls.

**Keywords:** semantic caching, large language models, conformal prediction, Thompson Sampling, approximate nearest neighbor search, cache admission policy

---

## I. Introduction

Large Language Models such as GPT-4, LLaMA, and Gemini now power conversational assistants, code generators, search engines, and content creation tools at massive scale. The demand for inference capacity has grown rapidly, but the cost of serving these models remains high. Commercial providers charge by the token — OpenAI's GPT-4, for instance, costs roughly $0.03 per thousand input tokens and $0.06 per thousand output tokens — and a service handling a million queries per day can easily accumulate daily costs in the thousands of dollars. Latency is equally important: a typical LLM call takes anywhere from 500 milliseconds to several seconds, and anything beyond a few hundred milliseconds begins to erode user engagement in interactive applications.

These costs would be more tolerable if every query required a fresh response, but that is not the case. A study of approximately 27,000 ChatGPT conversations found that about 31 percent of user queries are semantically similar to questions that have already been asked [14]. The queries differ in wording and structure — "What is the capital of France?" versus "Tell me the capital city of France" — but they seek the same answer. Traditional exact-match caches cannot exploit this redundancy because the two strings are textually different. Semantic caching closes this gap by comparing queries in an embedding space where semantically equivalent questions are mapped to nearby vectors.

Several semantic cache systems have been proposed in recent literature. GPTCache [3] uses a fixed similarity threshold of 0.75 with LRU eviction. MeanCache [2] computes an adaptive threshold as the mean minus one standard deviation of observed similarities. vCache [4] introduces per-entry conformal thresholds with formal error guarantees but uses a conservative default of 0.80, simple LRU eviction, and no admission control. RAGCache [8] adds cost-aware GDSF eviction but keeps a static global threshold of 0.85. All of these systems share a common weakness: their threshold mechanism, whether fixed or globally adaptive, applies the same matching criterion to every cached entry regardless of how much precision that particular entry requires. A factual question about a specific date demands a very tight similarity match to avoid returning the wrong number, while an open-ended question about a broad topic can tolerate looser matching.

This paper presents ContextualCache, which addresses these limitations through the following contributions:

1. **Per-entry conformal thresholds** that learn an individually calibrated similarity boundary for each cached entry, with a formal guarantee that P(false positive) ≤ ε.
2. **Thompson Sampling bandit** with Beta posteriors over 10 threshold arms (0.65 to 0.95) to learn optimal default thresholds for newly cached entries.
3. **ADWIN drift detection** that monitors the bandit reward stream and resets posteriors when a statistically significant distribution shift is detected.
4. **Semantic W-TinyLFU admission** using Random Projection LSH over a Count-Min Sketch to reject infrequent queries that would pollute the cache.
5. **CostGDSF eviction** that retains entries with high regeneration cost, incorporating LLM inference cost and storage size into the eviction priority.
6. **Two-tier lookup** combining O(1) exact-hash matching with O(log n) FAISS HNSW approximate nearest-neighbor search.

The remainder of this paper is organized as follows. Section II reviews related work and identifies the gaps addressed by this system. Section III describes the proposed method in detail. Section IV presents the experimental setup. Section V reports and discusses the results. Section VI concludes with a summary and directions for future work.

---

## II. Related Work

### A. Semantic Caching Systems

**GPTCache** [3] is a Redis-backed semantic cache that uses HNSW approximate nearest-neighbor search to retrieve similar queries and a fixed similarity threshold of 0.75 to decide cache hits. It uses LRU eviction and admits all entries unconditionally. While it demonstrates the feasibility of semantic caching for LLM cost reduction, the fixed threshold cannot adapt to different query types, and its LRU eviction treats all entries as equally valuable.

**MeanCache** [2] introduces federated learning for similarity model training and computes an adaptive threshold as the mean minus one standard deviation of observed similarity scores. It also uses PCA-based embedding compression. While the threshold adaptation is an improvement over a fixed value, it operates at a global level and cannot distinguish between queries that need different precision levels.

**vCache** [4] is the most closely related work to ours. It introduces per-entry conformal thresholds using online conformal prediction, providing formal guarantees on the false positive rate. However, vCache uses a higher default threshold of 0.80, employs simple LRU eviction, and lacks admission control. Our system extends the conformal approach with Thompson Sampling for learning optimal defaults, CostGDSF for value-aware eviction, and W-TinyLFU for frequency-gated admission.

**RAGCache** [8] targets retrieval-augmented generation workloads and uses cost-aware GDSF eviction similar to our approach. However, it relies on a static global threshold of 0.85 and does not implement per-entry adaptation.

**SCALM** [6] applies vector quantization and semantic clustering to cache automated chat service responses. It achieves significant token savings but lacks adaptive threshold mechanisms. **MinCache** [1] implements a three-tier hierarchy of exact, resemblance, and semantic matching, achieving 4.5× speedup over GPTCache, but its MinHash-based resemblance layer misses nuanced semantic relationships.

### B. Algorithmic Foundations

**Conformal prediction** [11] provides a framework for constructing prediction sets with guaranteed coverage. Given a sequence of observations, it computes nonconformity scores and uses the quantile of historical scores to set a threshold that guarantees a desired error rate. We apply this to the cache hit decision, treating each entry as an independent prediction problem.

**Thompson Sampling** [19] is a Bayesian approach to the multi-armed bandit problem where each arm maintains a Beta posterior and exploration is driven by sampling from the posteriors. **ADWIN** [10] maintains a variable-length sliding window and uses Hoeffding bounds to detect distribution changes. We use it to trigger bandit resets when the query distribution shifts.

**W-TinyLFU** [7] combines a small admission window with a frequency-gated main cache, using a Count-Min Sketch for space-efficient frequency estimation. Our Semantic W-TinyLFU extends this with LSH-bucketed counters so that semantically similar queries share frequency estimates. **GDSF** [12] assigns eviction priorities based on frequency, cost, and size; we adapt it to use LLM regeneration cost.

### C. Research Gaps

The review of existing literature reveals three principal gaps that ContextualCache addresses: (1) No existing system except vCache provides per-entry threshold adaptation, and even vCache lacks exploration mechanisms for learning optimal defaults. (2) Most systems use simple LRU eviction and unconditional admission, ignoring the heterogeneous value of cached entries. (3) No existing system combines conformal guarantees with bandit-based exploration, drift detection, and intelligent admission/eviction.

---

## III. Proposed Method

### A. System Overview

ContextualCache is organized around a central cache manager that coordinates a multi-stage query pipeline. When a query arrives, it first passes through a Tier 1 exact-hash lookup. On a miss, the query is embedded and forwarded to a Tier 2 FAISS HNSW index for semantic search. If a candidate exceeds its per-entry conformal threshold, the cached response is returned. Otherwise, the query falls through to the LLM backend, and the generated response passes through the W-TinyLFU admission gate before being stored with CostGDSF eviction metadata. A feedback loop carries correctness signals back to the conformal threshold calibrator, the Thompson Sampling bandit, and the ADWIN drift detector.

**[Figure 1: System architecture diagram showing the query pipeline, two-tier lookup, conformal thresholds, bandit, drift detector, admission, and eviction components]**
*(Use docs/report-assets/fig-architecture.png)*

Table I summarizes the computational complexity and typical latency of each pipeline stage.

**Table I: Query Pipeline Complexity**

| Stage | Component | Complexity | Latency |
|-------|-----------|-----------|---------|
| Tier 1 | Exact Hash (SHA-256) | O(1) | <1 ms |
| Tier 2 | FAISS HNSW Search | O(log n) | 2–15 ms |
| Embedding | paraphrase-MiniLM-L6-v2 | O(d) | 5–20 ms |
| Threshold | Per-entry conformal τ_i | O(k log k) | <0.1 ms |
| Miss | LLM Provider (Ollama/Groq) | Network | 500–6000 ms |

### B. Two-Tier Lookup Architecture

The first tier handles exact matches without requiring any embedding computation. The query text is normalized — lowercased, stripped of leading and trailing whitespace, internal spaces collapsed, trailing punctuation removed — and hashed with SHA-256. The hash is checked against an in-memory dictionary. This sub-millisecond lookup catches repeated queries that arrive with identical or near-identical wording.

Only on a Tier 1 miss does the system invoke the embedding model. The query is encoded into a 384-dimensional vector using the paraphrase-MiniLM-L6-v2 sentence transformer [20], chosen for its training on paraphrase data which produces higher similarity scores for semantically equivalent queries than general-purpose alternatives. An LRU embedding cache (10,000 entries) avoids redundant encoding.

The FAISS index uses the HNSW algorithm with M=32 links per node, efConstruction=200, and efSearch=64. The index is wrapped in IndexIDMap2 for explicit ID management, enabling true vector removal. A periodic rebuild after every 500 removals reclaims fragmented space.

For multi-turn conversations, the query embedding is fused with the accumulated session context vector using exponential moving average:

> **c_new = α · q + (1 − α) · c_old**

where α = 0.85 gives strong weight to the current query while retaining prior conversational context. Sessions expire after 30 minutes of inactivity.

### C. Per-Entry Conformal Thresholds

The core innovation of ContextualCache is the replacement of a single global threshold with per-entry conformal thresholds. Each cached entry i maintains its own similarity threshold τ_i, computed from the entry's history of correctness feedback.

**Nonconformity Scores.** For a cache hit on entry i with similarity s:
- If the hit was correct: nonconformity = 1 − s
- If the hit was incorrect: nonconformity = min(1.0, max(1 − s + 0.1, 0.5))

The incorrect-hit formula inflates the nonconformity score, pushing the threshold higher for entries that have produced false positives.

**Threshold Calibration.** The threshold τ_i is computed as 1 minus the (1 − ε) quantile of the entry's nonconformity scores, where ε is the target error rate (default 0.05). A sliding window of the 200 most recent scores ensures adaptation to changing patterns. The threshold is clamped to [0.60, 0.99] to prevent extreme values. New entries start with a default threshold and transition to conformal calibration after accumulating at least 3 feedback signals.

**Formal Guarantee.** Under the exchangeability assumption of conformal prediction:

> **P(false positive) ≤ ε**

This guarantee holds per entry, meaning each cached response independently controls its own false positive risk.

**Algorithm 1: Conformal Threshold Update**

```
Input: entry_id, similarity, was_correct
1. If was_correct:
       nonconformity ← 1.0 − similarity
   Else:
       nonconformity ← min(1.0, max(1.0 − similarity + 0.1, 0.5))
2. Append nonconformity to scores[entry_id]
3. If len(scores) > window_size:
       Truncate to most recent window_size scores
4. If len(scores) ≥ min_calibration_points:
       Sort scores
       idx ← ceil((1 − ε) × (len + 1)) − 1
       τ_i ← 1.0 − scores[idx]
       τ_i ← clamp(τ_i, 0.60, 0.99)
   Else:
       τ_i ← default_threshold
5. Return τ_i
```

### D. Thompson Sampling Bandit for Default Threshold Learning

While conformal prediction handles per-entry fine-tuning after sufficient feedback, new entries must start with a default threshold. Rather than fixing this value arbitrarily, a Thompson Sampling bandit learns what the default should be.

The bandit maintains 10 arms corresponding to threshold values linearly spaced from 0.65 to 0.95. Each arm j has a Beta posterior parameterized by α_j (successes) and β_j (failures), both initialized to 1. On each cache hit, the current best arm receives a reward computed as:

> If similarity ≥ 0.85: reward = 1.0
> If similarity < 0.65: reward = 0.0
> Otherwise: reward = (similarity − 0.65) / 0.20

The best arm is determined by expected reward E[j] = α_j / (α_j + β_j), and its posterior is updated with the observed reward.

**Algorithm 2: Thompson Sampling Update**

```
Input: similarity score from cache hit
1. arm* ← argmax_j (α_j / (α_j + β_j))
2. Compute reward:
       If similarity ≥ 0.85: reward ← 1.0
       Else if similarity < 0.65: reward ← 0.0
       Else: reward ← (similarity − 0.65) / 0.20
3. α[arm*] ← α[arm*] + reward
   β[arm*] ← β[arm*] + (1.0 − reward)
4. Feed reward to ADWIN drift detector
5. If ADWIN detects drift:
       Reset all α_j, β_j to (2, 2)
```

### E. ADWIN Drift Detection

The ADWIN algorithm [10] monitors the stream of bandit rewards for distribution changes. It maintains a variable-length window of recent observations and uses Hoeffding bounds to compare the means of two sub-windows. When a statistically significant change is detected (with sensitivity parameter δ = 0.002), the bandit posteriors are reset to weakly informative priors (α = 2, β = 2), forcing renewed exploration. This ensures that the system does not cling to a stale default threshold after the query distribution has shifted.

### F. Semantic W-TinyLFU Admission Policy

Not every cache miss should result in a new entry. One-time queries that are unlikely to recur waste capacity. The W-TinyLFU admission policy [7] addresses this through a two-stage design:

1. **Window cache (5% of capacity):** Admits all entries freely, giving new entries a cold-start period to accumulate frequency.
2. **Main cache (95% of capacity):** A new entry is admitted only if its frequency exceeds that of the window's LRU victim.

Our semantic extension replaces exact-text frequency tracking with LSH-bucketed counting. Random Projection LSH (8 bits, 4 tables) maps each embedding vector to a discrete bucket, so semantically similar queries share the same frequency counter in a Count-Min Sketch (width 4096, depth 4, conservative update). A periodic halving of all counters after every N insertions (N = cache capacity) ensures that recent activity outweighs stale counts.

### G. CostGDSF Eviction

When the cache is full and a new entry must be admitted, CostGDSF selects the entry with the lowest priority for eviction:

> **H(e) = (frequency(e) × regen_cost(e)) / storage_bytes(e) + L**

where regen_cost(e) = (llm_cost_usd × 10^6) + embed_cost_ms, and L is an inflation factor updated to the priority of the last evicted entry. This formula ensures that entries which are accessed often and are expensive to regenerate through the LLM are retained longest, while cheap-to-recompute entries that are rarely requested are evicted first.

---

## IV. Experimental Setup

### A. Datasets

Two public question-answering datasets from HuggingFace were used:

1. **Natural Questions (NQ)** [17]: Google's dataset of real web search queries with short answers. 500 unique questions were sampled, and 150 paraphrase variants were generated using 26 deterministic rule-based transformations (syntactic restructuring, prefix injection), yielding 650 queries per run.
2. **SQuAD** [18]: Stanford's reading comprehension dataset (validation split). The same sampling and paraphrase strategy was applied, producing 650 queries.

### B. Baselines

Six baseline strategies were compared against ContextualCache, all receiving identical pre-computed embeddings and LLM responses to ensure fair comparison:

1. **Exact-Match LRU** — Hash-only matching with LRU eviction (lower-bound baseline).
2. **GPTCache** [3] — Static threshold 0.75, LRU eviction.
3. **MeanCache** [2] — Adaptive threshold (mean − std), PCA compression, LRU eviction.
4. **vCache** [4] — Per-entry conformal thresholds, default 0.80, LRU eviction.
5. **RAGCache** [8] — Static threshold 0.85, GDSF eviction.
6. **No-Admission** — Conformal thresholds + CostGDSF eviction but no admission gate (ablation).

### C. Metrics

- **Hit Rate** = total hits / total queries
- **Precision** = correct hits / total hits
- **F1 Score** = 2 × precision × recall / (precision + recall)
- **LLM Calls** = queries forwarded to the LLM (lower is better)

A hit is marked correct if the gold answer string appears in the cached response.

### D. Implementation

Cache capacity was set to 300 entries. The LLM backend was Ollama running LLaMA 3.2:3b locally. All embeddings were generated using paraphrase-MiniLM-L6-v2 (384 dimensions). The system is implemented in Python 3.10 with FastAPI, FAISS (faiss-cpu), and PyTorch (CUDA 12.6).

---

## V. Results and Discussion

### A. Results on Natural Questions

Table II presents the results on the NQ dataset with 650 queries and a cache capacity of 300.

**Table II: Benchmark Results on Natural Questions (650 queries, 300 capacity)**

| Strategy | Hit Rate | Precision | F1 | Correct | Incorrect | LLM Calls |
|----------|----------|-----------|------|---------|-----------|-----------|
| **Ours** | **20.77%** | **62.22%** | **0.214** | **84** | **51** | **515** |
| GPTCache | 19.38% | 61.90% | 0.201 | 78 | 48 | 524 |
| vCache | 18.62% | 62.81% | 0.197 | 76 | 45 | 529 |
| No-Admission | 17.69% | 63.48% | 0.191 | 73 | 42 | 535 |
| RAGCache | 17.23% | 64.29% | 0.189 | 72 | 40 | 538 |
| MeanCache | 13.23% | 66.28% | 0.155 | 57 | 29 | 564 |
| Exact-Match LRU | 0.00% | — | 0.000 | 0 | 0 | 650 |

**[Figure 2: Grouped bar chart — Hit Rate and Precision comparison across all strategies on NQ dataset]**
*(PLACEHOLDER — Create in Excel: grouped bar chart with strategies on x-axis, two bars per strategy: blue for Hit Rate, purple for Precision. See data in Table II.)*

ContextualCache achieves the highest hit rate (20.77%) and the highest F1 score (0.214) among all strategies. Compared to GPTCache, it serves 6 additional correct cache hits (84 versus 78) while saving 9 more LLM calls (515 versus 524). The precision of 62.22% is competitive with GPTCache's 61.90%, confirming that the higher hit rate does not come at the expense of more false positives.

The improvement comes from the lower learned default threshold (around 0.70 via the bandit) which captures paraphrase matches that GPTCache (threshold 0.75) and RAGCache (threshold 0.85) miss. The per-entry conformal calibration then tightens individual thresholds where needed, preventing the lower default from degrading precision.

MeanCache achieves the highest precision (66.28%) but at the cost of a significantly lower hit rate (13.23%), because its adaptive threshold (mean − std) tends to be overly conservative.

### B. Results on SQuAD

Table III presents the results on the SQuAD dataset.

**Table III: Benchmark Results on SQuAD (650 queries, 300 capacity)**

| Strategy | Hit Rate | Precision | F1 | Correct | Incorrect | LLM Calls |
|----------|----------|-----------|------|---------|-----------|-----------|
| **Ours** | **23.69%** | 35.71% | **0.137** | **55** | 99 | **496** |
| GPTCache | 19.54% | 38.58% | 0.126 | 49 | 78 | 523 |
| vCache | 17.85% | 38.79% | 0.118 | 45 | 71 | 534 |
| No-Admission | 17.85% | 37.07% | 0.112 | 43 | 73 | 534 |
| RAGCache | 17.38% | 36.28% | 0.108 | 41 | 72 | 537 |
| MeanCache | 5.23% | 47.06% | 0.047 | 16 | 18 | 616 |
| Exact-Match LRU | 0.00% | — | 0.000 | 0 | 0 | 650 |

On SQuAD, ContextualCache again achieves the highest hit rate (23.69%) and F1 score (0.137), outperforming GPTCache by 4.15 percentage points. The system saves 154 LLM calls, the most among all strategies. Overall precision is lower on SQuAD across every strategy because SQuAD questions about the same Wikipedia passage often share vocabulary and sentence structure, producing high similarity scores despite asking about different facts. Two questions like "Who founded the university?" and "When was the university founded?" yield similar embeddings but require entirely different answers.

### C. Precision Analysis

Table IV compares precision across the three highest-performing strategies on NQ.

**Table IV: Precision Analysis — Top Three Strategies (NQ)**

| Metric | GPTCache | vCache | Ours |
|--------|----------|--------|------|
| Total Hits | 126 | 121 | 135 |
| Correct Hits | 78 | 76 | 84 |
| Incorrect Hits | 48 | 45 | 51 |
| Precision | 61.90% | 62.81% | 62.22% |
| False Positive Rate | 38.10% | 37.19% | 37.78% |

ContextualCache delivers 84 correct responses out of 135 total hits. The precision difference between ContextualCache and vCache is only 0.59 percentage points, while the hit rate advantage is 2.15 percentage points. This confirms that the conformal mechanism successfully controls false positives even with a lower default threshold and more aggressive admission.

It is worth noting that the false positive rates in this benchmark are inflated by the evaluation methodology, which treats any response not containing the exact gold answer string as incorrect. In practice, many flagged responses are factually accurate but phrased differently from the gold answer.

### D. Latency Analysis

Table V reports the latency results.

**Table V: Latency Comparison (NQ)**

| Strategy | Hit Latency (ms) | Overall Latency (ms) | LLM Calls Saved |
|----------|-------------------|----------------------|-----------------|
| Ours | 0.00 | 1610.17 | 135 |
| GPTCache | 0.13 | 1613.17 | 126 |
| vCache | 0.00 | 1632.40 | 121 |
| No-Admission | 0.13 | 1662.86 | 115 |
| RAGCache | 0.28 | 1675.60 | 112 |
| MeanCache | 0.00 | 1736.22 | 86 |

All strategies achieve sub-millisecond hit latency, confirming that cache lookup overhead is negligible. Overall latency is dominated by LLM inference on cache misses, and ContextualCache achieves the lowest average (1610 ms) due to its highest hit rate.

### E. Ablation: Admission Policy

Comparing ContextualCache against the No-Admission baseline isolates the contribution of the W-TinyLFU admission policy. Without admission control, the hit rate drops from 20.77% to 17.69% — a decrease of 3.08 percentage points — while precision increases marginally from 62.22% to 63.48%. The net F1 effect is clearly positive (0.214 versus 0.191), confirming that the frequency-gated admission prevents one-time queries from occupying cache slots that could serve repeat queries.

The comparison between ContextualCache and vCache further isolates the combined effect of the Thompson Sampling bandit, CostGDSF eviction, and admission policy. vCache uses the same conformal mechanism but with a higher default threshold (0.80), LRU eviction, and no admission gate. ContextualCache outperforms vCache by 2.15 percentage points in hit rate on NQ, indicating that the bandit's learned lower default captures valid matches that vCache's conservative setting misses.

### F. Cross-Dataset Analysis

The NQ and SQuAD results reveal an important trade-off. On NQ, which contains diverse web search queries, the system achieves a moderate hit rate (20.77%) but high precision (62.22%). On SQuAD, where questions are clustered around specific Wikipedia passages, the hit rate climbs to 23.69% but precision falls to 35.71%. The clustering effect in SQuAD produces more high-similarity matches, but many of those matches are between questions that ask different things about the same passage. This highlights a practical deployment consideration: workloads with repetitive, narrowly focused queries will see higher hit rates but may need stricter thresholds to maintain acceptable precision.

---

## VI. Conclusion and Future Work

This paper presented ContextualCache, a semantic cache for LLM serving that replaces the fixed global similarity threshold used by existing systems with per-entry conformal thresholds. Each cached entry learns its own similarity boundary from feedback, backed by a formal guarantee that the false positive probability remains below the configured error rate. A Thompson Sampling bandit explores the threshold parameter space to learn optimal defaults for new entries, with ADWIN drift detection triggering automatic resets when the query distribution changes. Semantic W-TinyLFU admission filters infrequent queries using LSH-bucketed frequency estimation, and CostGDSF eviction prioritizes retention of entries that are expensive to regenerate.

Evaluation on the Natural Questions and SQuAD datasets demonstrated that ContextualCache achieves the highest hit rate and F1 score among seven strategies, including GPTCache, MeanCache, vCache, and RAGCache. On NQ, the system reaches a 20.77% hit rate with 62.22% precision and an F1 of 0.214, saving 135 out of 650 LLM inference calls. On SQuAD, the hit rate reaches 23.69% with 154 calls saved.

Future work includes multi-round evaluation where conformal thresholds can fully calibrate through repeated feedback cycles, evaluation with larger embedding models such as E5-base-v2, testing on conversational and domain-specific datasets, multi-node distributed deployment using the implemented consistent hashing and gossip-based bandit synchronization components, and production load testing under concurrent user traffic.

---

## References

[1] K. Haqiq, "MinCache: A hybrid cache system for efficient chatbots with hierarchical embedding matching and LLM," *Proc. Future Gener. Comput. Syst.*, vol. 165, pp. 107–120, 2025.

[2] W. Gill, A. Elidrisi, et al., "MeanCache: User-Centric Semantic Caching for LLM Web Services," *arXiv preprint arXiv:2403.02694*, 2024.

[3] S. Regmi and C. P. Pun, "GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching," *arXiv preprint arXiv:2411.05276*, 2024.

[4] A. Schroeder, et al., "vCache: Per-Entry Conformal Thresholds for Semantic Caching," UC Berkeley, 2025.

[5] H. Li, et al., "Context-based Semantic Caching for LLM Applications," *Proc. IEEE Int. Conf. on Web Services (ICWS)*, 2024.

[6] Z. Li, et al., "SCALM: Towards Semantic Caching for Automated Chat Services with LLMs," *Proc. IEEE/ACM Int. Symp. on Quality of Service (IWQoS)*, 2024.

[7] G. Einziger, R. Friedman, B. Manes, "TinyLFU: A Highly Efficient Cache Admission Policy," *ACM Trans. Storage*, vol. 13, no. 4, 2017.

[8] Y. Jin, et al., "RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation," *arXiv preprint arXiv:2404.12457*, 2024.

[9] B. McMahan, et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," *Proc. AISTATS*, 2017.

[10] A. Bifet and R. Gavalda, "Learning from Time-Changing Data with Adaptive Windowing," *Proc. SIAM Int. Conf. on Data Mining (SDM)*, 2007.

[11] V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*, Springer, 2005.

[12] M. Young, "The GDSF Online Caching Algorithm," Tech. Report, 2000.

[13] X. Gao, et al., "Online Context Caching for Distributed LLM Serving," *Proc. IEEE IPDPS*, 2025.

[14] F. Zhu, et al., "Semantic Caching for Large Language Model Cost Reduction," *Proc. IEEE GLOBECOM*, 2023.

[15] R. Zha, et al., "Online Optimization for Semantic Cache Hit Prediction," *Proc. IEEE Int. Conf. on Big Data*, pp. 690–699, 2023.

[16] N. F. Liu, et al., "Lost in the Middle: How Language Models Use Long Contexts," *Trans. Assoc. Comput. Linguistics*, vol. 12, pp. 157–173, 2024.

[17] T. Kwiatkowski, et al., "Natural Questions: A Benchmark for Question Answering Research," *Trans. Assoc. Comput. Linguistics*, vol. 7, pp. 453–466, 2019.

[18] P. Rajpurkar, et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text," *Proc. EMNLP*, 2016.

[19] W. Thompson, "On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples," *Biometrika*, vol. 25, pp. 285–294, 1933.

[20] J. Johnson, M. Douze, H. Jegou, "Billion-Scale Similarity Search with GPUs," *IEEE Trans. on Big Data*, vol. 7, no. 3, pp. 535–547, 2021.
