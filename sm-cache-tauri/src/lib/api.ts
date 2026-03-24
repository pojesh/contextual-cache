/**
 * API client for the ContextualCache backend.
 * All calls go to the FastAPI server.
 */

const BASE_URL = 'http://localhost:8000';

export interface QueryRequest {
  query: string;
  session_id?: string;
  tenant_id?: string;
}

export interface QueryResponse {
  response: string;
  hit: boolean;
  tier: number;
  latency_ms: number;
  entry_id?: string;
  similarity?: number;
  admitted?: boolean;
  llm_latency_ms?: number;
  error?: boolean;
  fallback?: boolean;
}

export interface FeedbackRequest {
  entry_id: string;
  was_correct: boolean;
  similarity?: number;
}

export interface AggregateMetrics {
  total_queries: number;
  total_hits: number;
  tier1_hits: number;
  tier2_hits: number;
  total_misses: number;
  total_evictions: number;
  total_admissions: number;
  admission_rejections: number;
  hit_rate: number;
  tier1_rate: number;
  tier2_rate: number;
  false_hit_rate: number;
  precision: number;
  avg_latency_ms: number;
  avg_hit_latency_ms: number;
  avg_miss_latency_ms: number;
  cache_size: number;
  cache_capacity: number;
  correct_hits: number;
  incorrect_hits: number;
}

export interface TimeSeriesPoint {
  timestamp: number;
  hit: boolean;
  tier: number;
  latency_ms: number;
  similarity: number;
  hit_rate: number;
}

export interface LatencyDistribution {
  p50: number;
  p90: number;
  p95: number;
  p99: number;
}

export interface BanditStats {
  shard_id: string;
  n_arms: number;
  total_updates: number;
  drift_resets: number;
  best_arm: number;
  best_threshold: number;
  arm_thresholds: number[];
  arm_expected_rewards: number[];
  arm_alphas: number[];
  arm_betas: number[];
}

export interface FullStats {
  aggregate: AggregateMetrics;
  time_series: TimeSeriesPoint[];
  latency_distribution: LatencyDistribution;
  threshold_distribution: number[];
  similarity_distribution: number[];
  eviction: Record<string, any>;
  admission: Record<string, any>;
  bandit: BanditStats;
  thresholds: Record<string, any>;
  lookup_engine: Record<string, any>;
  embedding_service: Record<string, any>;
  llm: Record<string, any>;
  sessions: Record<string, any>;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function sendQuery(req: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>('/query', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function sendFeedback(req: FeedbackRequest): Promise<any> {
  return request('/feedback', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function getStats(): Promise<FullStats> {
  return request<FullStats>('/api/stats');
}

export async function getAnalytics(): Promise<any> {
  return request('/api/analytics');
}

export async function getHealth(): Promise<any> {
  return request('/health');
}

export async function clearCache(): Promise<any> {
  return request('/api/clear-cache', { method: 'POST' });
}

// ── Benchmark API ────────────────────────────────────────────────

export interface BenchmarkRunRequest {
  num_questions: number;
  capacity: number;
}

export interface BenchmarkStrategyResult {
  name: string;
  total_queries: number;
  total_hits: number;
  total_misses: number;
  hit_rate: number;
  precision: number;
  f1_score: number;
  avg_hit_latency_ms: number;
  avg_miss_latency_ms: number;
  avg_overall_latency_ms: number;
  llm_calls: number;
  correct_hits: number;
  incorrect_hits: number;
  tier1_hits: number;
  tier2_hits: number;
  total_time_s: number;
}

export interface BenchmarkResult {
  run_id: string;
  timestamp: number;
  num_questions: number;
  num_paraphrases: number;
  cache_capacity: number;
  strategies: BenchmarkStrategyResult[];
}

export interface BenchmarkProgress {
  run_id: string;
  status: string;
  total_strategies: number;
  completed_strategies: number;
  current_strategy: string;
  total_queries: number;
  completed_queries: number;
  error: string;
  elapsed_s: number;
}

export async function startBenchmark(req: BenchmarkRunRequest): Promise<{ status: string; run_id: string | null }> {
  return request('/api/benchmark/run', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function getBenchmarkStatus(runId: string): Promise<BenchmarkProgress> {
  return request<BenchmarkProgress>(`/api/benchmark/status/${runId}`);
}

export async function listBenchmarkResults(): Promise<{ results: any[] }> {
  return request('/api/benchmark/results');
}

export async function getBenchmarkResult(runId: string): Promise<BenchmarkResult> {
  return request<BenchmarkResult>(`/api/benchmark/results/${runId}`);
}
