<script module lang="ts">
  import type { QueryResponse, DirectQueryResponse } from '$lib/api';

  interface ComparisonEntry {
    id: string;
    query: string;
    cached: QueryResponse | null;
    direct: DirectQueryResponse | null;
    timeSaved: number;
    tokensSaved: number;
  }

  // Preserved state across page navigations
  let comparisons = $state<ComparisonEntry[]>([]);
  let input = $state('');
  let sessionId = $state('compare-' + Math.random().toString(36).slice(2, 8));
</script>

<script lang="ts">
  import { sendQuery, sendDirectQuery } from '$lib/api';

  let sending = $state(false);

  // Cumulative stats
  let totalQueries = $derived(comparisons.length);
  let totalTimeSaved = $derived(comparisons.reduce((s, c) => s + c.timeSaved, 0));
  let totalTokensSaved = $derived(comparisons.reduce((s, c) => s + c.tokensSaved, 0));
  let avgSpeedup = $derived(
    totalQueries > 0
      ? comparisons.reduce((s, c) => {
          if (c.cached && c.direct && c.direct.latency_ms > 0)
            return s + c.direct.latency_ms / Math.max(c.cached.latency_ms, 0.1);
          return s;
        }, 0) / totalQueries
      : 0
  );

  async function handleSend() {
    const text = input.trim();
    if (!text || sending) return;
    input = '';
    sending = true;

    const entry: ComparisonEntry = {
      id: crypto.randomUUID(),
      query: text,
      cached: null,
      direct: null,
      timeSaved: 0,
      tokensSaved: 0,
    };

    // Add to top of list immediately
    comparisons = [entry, ...comparisons];

    try {
      const [cachedRes, directRes] = await Promise.all([
        sendQuery({ query: text, session_id: sessionId }),
        sendDirectQuery(text),
      ]);

      entry.cached = cachedRes;
      entry.direct = directRes;
      entry.timeSaved = directRes.latency_ms - cachedRes.latency_ms;
      entry.tokensSaved = (directRes.total_tokens ?? 0) - (cachedRes.total_tokens ?? 0);

      comparisons = [entry, ...comparisons.slice(1)];
    } catch (e: any) {
      console.error('Comparison failed', e);
    } finally {
      sending = false;
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function formatMs(ms: number): string {
    if (ms >= 1000) return (ms / 1000).toFixed(2) + 's';
    return ms.toFixed(0) + 'ms';
  }

  function newComparison() {
    comparisons = [];
    sessionId = 'compare-' + Math.random().toString(36).slice(2, 8);
    input = '';
  }
</script>

<svelte:head>
  <title>Compare — ContextualCache</title>
</svelte:head>

<div class="compare-page">
  <!-- Cumulative Stats -->
  <header class="stats-bar">
    <div class="stat-card">
      <span class="stat-label">Queries</span>
      <span class="stat-num">{totalQueries}</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">Time Saved</span>
      <span class="stat-num accent-green">{formatMs(totalTimeSaved)}</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">Tokens Saved</span>
      <span class="stat-num accent-amber">{totalTokensSaved}</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">Avg Speedup</span>
      <span class="stat-num accent-indigo">{avgSpeedup.toFixed(1)}x</span>
    </div>
    <button class="new-btn" onclick={newComparison} title="New Comparison">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>
      New
    </button>
  </header>

  <!-- Input -->
  <div class="input-section">
    <input
      type="text"
      class="compare-input"
      placeholder="Type a query to compare cached vs direct LLM..."
      bind:value={input}
      onkeydown={handleKeydown}
      disabled={sending}
    />
    <button class="compare-btn" onclick={handleSend} disabled={sending || !input.trim()}>
      {sending ? 'Comparing...' : 'Compare'}
    </button>
  </div>

  <!-- Results -->
  <div class="results-scroll">
    {#each comparisons as comp (comp.id)}
      <div class="comparison-card">
        <div class="comp-query">
          <span class="comp-query-label">Query:</span>
          <span class="comp-query-text">{comp.query}</span>
        </div>

        <div class="comp-panels">
          <!-- Cached Panel -->
          <div class="panel panel-cached">
            <div class="panel-header">
              <span class="panel-title">With Cache</span>
              {#if comp.cached}
                <span class="panel-badge" class:hit={comp.cached.hit} class:miss={!comp.cached.hit}>
                  {comp.cached.hit ? `HIT · Tier ${comp.cached.tier}` : 'MISS'}
                </span>
              {/if}
            </div>
            {#if comp.cached}
              <div class="panel-meta">
                <span class="meta-chip latency">{formatMs(comp.cached.latency_ms)}</span>
                {#if comp.cached.hit && (comp.cached.similarity ?? 0) > 0}
                  <span class="meta-chip sim">sim {(comp.cached.similarity ?? 0).toFixed(3)}</span>
                {/if}
                <span class="meta-chip tokens">{comp.cached.total_tokens ?? 0} tok</span>
              </div>
              <p class="panel-response">{comp.cached.response}</p>
            {:else}
              <div class="panel-loading">Loading...</div>
            {/if}
          </div>

          <!-- Direct Panel -->
          <div class="panel panel-direct">
            <div class="panel-header">
              <span class="panel-title">Without Cache</span>
              <span class="panel-badge direct">DIRECT LLM</span>
            </div>
            {#if comp.direct}
              <div class="panel-meta">
                <span class="meta-chip latency">{formatMs(comp.direct.latency_ms)}</span>
                <span class="meta-chip tokens">{comp.direct.total_tokens} tok</span>
              </div>
              <p class="panel-response">{comp.direct.response}</p>
            {:else}
              <div class="panel-loading">Loading...</div>
            {/if}
          </div>
        </div>

        <!-- Delta Bar -->
        {#if comp.cached && comp.direct}
          <div class="delta-bar">
            <span class="delta" class:positive={comp.timeSaved > 0} class:negative={comp.timeSaved <= 0}>
              {comp.timeSaved > 0 ? '▲' : '▼'} {Math.abs(comp.timeSaved).toFixed(0)}ms
              {comp.timeSaved > 0 ? 'faster' : 'slower'}
            </span>
            {#if comp.tokensSaved > 0}
              <span class="delta positive">▲ {comp.tokensSaved} tokens saved</span>
            {/if}
            {#if comp.cached.hit && comp.direct.latency_ms > 0}
              <span class="delta speedup">
                {(comp.direct.latency_ms / Math.max(comp.cached.latency_ms, 0.1)).toFixed(1)}x speedup
              </span>
            {/if}
          </div>
        {/if}
      </div>
    {/each}

    {#if comparisons.length === 0}
      <div class="empty-state">
        <div class="empty-icon">⚡</div>
        <h2>Side-by-Side Comparison</h2>
        <p>Send the same query to both the semantic cache and the LLM directly. See the latency difference, token savings, and cache hit behavior in real time.</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .compare-page {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #0c0c14;
    padding: 0;
  }

  /* ── Stats Bar ──────────────────────────────────────────── */
  .stats-bar {
    display: flex;
    gap: 12px;
    padding: 16px 24px;
    border-bottom: 1px solid rgba(99, 102, 241, 0.08);
    background: rgba(12, 12, 20, 0.9);
    backdrop-filter: blur(12px);
    flex-shrink: 0;
  }

  .stat-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 20px;
    background: rgba(15, 15, 26, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.08);
    border-radius: 10px;
    min-width: 100px;
  }

  .stat-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #475569;
    font-weight: 600;
  }

  .stat-num {
    font-size: 18px;
    font-weight: 700;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
  }

  .accent-green { color: #34d399; }
  .accent-amber { color: #f59e0b; }
  .accent-indigo { color: #818cf8; }

  .new-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 8px;
    color: #94a3b8;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .new-btn:hover {
    background: rgba(99, 102, 241, 0.15);
    color: #c7d2fe;
  }

  /* ── Input Section ──────────────────────────────────────── */
  .input-section {
    display: flex;
    gap: 10px;
    padding: 16px 24px;
    flex-shrink: 0;
  }

  .compare-input {
    flex: 1;
    padding: 12px 16px;
    background: #151520;
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 12px;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
    font-size: 13.5px;
    outline: none;
    transition: border-color 0.2s;
  }

  .compare-input:focus {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08);
  }

  .compare-input::placeholder { color: #3b3d56; }
  .compare-input:disabled { opacity: 0.5; }

  .compare-btn {
    padding: 12px 24px;
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    border: none;
    border-radius: 12px;
    color: white;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .compare-btn:hover:not(:disabled) {
    transform: scale(1.02);
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
  }

  .compare-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ── Results ────────────────────────────────────────────── */
  .results-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 0 24px 24px;
  }

  .results-scroll::-webkit-scrollbar { width: 6px; }
  .results-scroll::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.15); border-radius: 3px; }

  .comparison-card {
    background: #111119;
    border: 1px solid rgba(99, 102, 241, 0.08);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .comp-query {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(99, 102, 241, 0.06);
  }

  .comp-query-label {
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    flex-shrink: 0;
  }

  .comp-query-text {
    font-size: 13px;
    color: #c7d2fe;
    font-weight: 500;
  }

  .comp-panels {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .panel {
    background: rgba(15, 15, 26, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.06);
    border-radius: 10px;
    padding: 12px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .panel-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
  }

  .panel-badge {
    font-size: 9px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 5px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }

  .panel-badge.hit {
    background: rgba(99, 102, 241, 0.12);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.2);
  }

  .panel-badge.miss {
    background: rgba(245, 158, 11, 0.1);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.15);
  }

  .panel-badge.direct {
    background: rgba(100, 116, 139, 0.12);
    color: #94a3b8;
    border: 1px solid rgba(100, 116, 139, 0.2);
  }

  .panel-meta {
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }

  .meta-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(255,255,255,0.03);
  }

  .meta-chip.latency { color: #34d399; }
  .meta-chip.sim { color: #a78bfa; }
  .meta-chip.tokens { color: #f59e0b; }

  .panel-response {
    font-size: 12.5px;
    line-height: 1.6;
    color: #94a3b8;
    margin: 0;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
  }

  .panel-loading {
    font-size: 12px;
    color: #475569;
    padding: 20px 0;
    text-align: center;
  }

  /* ── Delta Bar ──────────────────────────────────────────── */
  .delta-bar {
    display: flex;
    gap: 12px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(99, 102, 241, 0.06);
    flex-wrap: wrap;
  }

  .delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 6px;
  }

  .delta.positive {
    color: #34d399;
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.15);
  }

  .delta.negative {
    color: #f87171;
    background: rgba(248, 113, 113, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.15);
  }

  .delta.speedup {
    color: #818cf8;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.15);
  }

  /* ── Empty State ────────────────────────────────────────── */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 24px;
    text-align: center;
  }

  .empty-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.7; }

  .empty-state h2 {
    font-size: 20px;
    font-weight: 700;
    color: #c7d2fe;
    margin-bottom: 8px;
  }

  .empty-state p {
    font-size: 13px;
    color: #64748b;
    max-width: 500px;
    line-height: 1.6;
  }
</style>
