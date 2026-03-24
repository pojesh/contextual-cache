<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart, registerables } from 'chart.js';
  import { getStats, clearCache } from '$lib/api';
  import type { FullStats } from '$lib/api';

  Chart.register(...registerables);

  let stats: FullStats | null = $state(null);
  let refreshInterval: ReturnType<typeof setInterval>;

  let hitRateChart: Chart | null = null;
  let latencyChart: Chart | null = null;
  let tierChart: Chart | null = null;
  let banditChart: Chart | null = null;
  let similarityChart: Chart | null = null;
  let thresholdChart: Chart | null = null;

  async function fetchStats() {
    try { stats = await getStats(); updateCharts(); } catch { /* */ }
  }

  async function handleClear() {
    try { await clearCache(); await fetchStats(); } catch { /* */ }
  }

  // ── Chart helpers ──────────────────────────────────────────
  const GRID = 'rgba(148,163,184,0.06)';
  const TICK = { family: 'Inter', size: 10 };

  function lineOpts(): any {
    return {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: GRID }, ticks: { color: '#475569', font: TICK, maxTicksLimit: 8 } },
        y: { grid: { color: GRID }, ticks: { color: '#475569', font: TICK } },
      },
    };
  }
  function barOpts(): any {
    return {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#475569', font: TICK, maxTicksLimit: 10 } },
        y: { grid: { color: GRID }, ticks: { color: '#475569', font: TICK } },
      },
    };
  }

  function updateCharts() {
    if (!stats) return;
    const ts = stats.time_series;
    const labels = ts.map((_: any, i: number) => String(i));

    // Hit Rate
    const hrCtx = document.getElementById('hr') as HTMLCanvasElement;
    if (hrCtx && ts.length) {
      const data = ts.map((p: any) => p.hit_rate * 100);
      if (hitRateChart) { hitRateChart.data.labels = labels; hitRateChart.data.datasets[0].data = data; hitRateChart.update('none'); }
      else hitRateChart = new Chart(hrCtx, { type: 'line', data: { labels, datasets: [{ data, borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.08)', fill: true, tension: 0.4, borderWidth: 2, pointRadius: 0 }] }, options: lineOpts() });
    }
    // Latency
    const ltCtx = document.getElementById('lt') as HTMLCanvasElement;
    if (ltCtx && ts.length) {
      const data = ts.map((p: any) => p.latency_ms);
      if (latencyChart) { latencyChart.data.labels = labels; latencyChart.data.datasets[0].data = data; latencyChart.update('none'); }
      else latencyChart = new Chart(ltCtx, { type: 'line', data: { labels, datasets: [{ data, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.08)', fill: true, tension: 0.4, borderWidth: 2, pointRadius: 0 }] }, options: lineOpts() });
    }
    // Tier Doughnut
    const tiCtx = document.getElementById('ti') as HTMLCanvasElement;
    if (tiCtx) {
      const d = [stats.aggregate.tier1_hits, stats.aggregate.tier2_hits, stats.aggregate.total_misses];
      if (tierChart) { tierChart.data.datasets[0].data = d; tierChart.update('none'); }
      else tierChart = new Chart(tiCtx, { type: 'doughnut', data: { labels: ['Tier 1 (Exact)', 'Tier 2 (Semantic)', 'Miss (LLM)'], datasets: [{ data: d, backgroundColor: ['#6366f1', '#8b5cf6', '#1e1b4b'], borderColor: ['#818cf8', '#a78bfa', '#312e81'], borderWidth: 2 }] }, options: { responsive: true, maintainAspectRatio: false, cutout: '60%', plugins: { legend: { position: 'bottom', labels: { color: '#64748b', font: { family: 'Inter', size: 11 }, padding: 12, usePointStyle: true, pointStyleWidth: 8 } } } } });
    }
    // Bandit
    const baCtx = document.getElementById('ba') as HTMLCanvasElement;
    if (baCtx && stats.bandit) {
      const armLabels = stats.bandit.arm_thresholds.map((t: number) => 'τ=' + t.toFixed(2));
      const data = stats.bandit.arm_expected_rewards;
      if (banditChart) { banditChart.data.labels = armLabels; banditChart.data.datasets[0].data = data; banditChart.update('none'); }
      else banditChart = new Chart(baCtx, { type: 'bar', data: { labels: armLabels, datasets: [{ data, backgroundColor: data.map((_: any, i: number) => i === stats!.bandit.best_arm ? '#6366f1' : 'rgba(99,102,241,0.18)'), borderRadius: 4 }] }, options: barOpts() });
    }
    // Similarity
    const smCtx = document.getElementById('sm') as HTMLCanvasElement;
    if (smCtx && stats.similarity_distribution.length) {
      const bins = Array(20).fill(0);
      stats.similarity_distribution.forEach((s: number) => bins[Math.min(Math.floor(s * 20), 19)]++);
      const binLabels = bins.map((_: number, i: number) => (i / 20).toFixed(2));
      if (similarityChart) { similarityChart.data.labels = binLabels; similarityChart.data.datasets[0].data = bins; similarityChart.update('none'); }
      else similarityChart = new Chart(smCtx, { type: 'bar', data: { labels: binLabels, datasets: [{ data: bins, backgroundColor: 'rgba(16,185,129,0.3)', borderColor: '#10b981', borderWidth: 1, borderRadius: 3 }] }, options: barOpts() });
    }
    // Thresholds
    const thCtx = document.getElementById('th') as HTMLCanvasElement;
    if (thCtx && stats.threshold_distribution.length) {
      const thLabels = stats.threshold_distribution.map((_: number, i: number) => String(i));
      if (thresholdChart) { thresholdChart.data.labels = thLabels; thresholdChart.data.datasets[0].data = stats.threshold_distribution; thresholdChart.update('none'); }
      else thresholdChart = new Chart(thCtx, { type: 'line', data: { labels: thLabels, datasets: [{ data: stats.threshold_distribution, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.08)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 0 }] }, options: lineOpts() });
    }
  }

  onMount(() => { fetchStats(); refreshInterval = setInterval(fetchStats, 3000); });
  onDestroy(() => { clearInterval(refreshInterval); [hitRateChart,latencyChart,tierChart,banditChart,similarityChart,thresholdChart].forEach(c => c?.destroy()); });
</script>

<svelte:head>
  <title>Analytics — ContextualCache</title>
</svelte:head>

<div class="analytics">
  <header class="topbar">
    <h1 class="page-title">Analytics</h1>
    <button class="btn-clear" onclick={handleClear}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
      Clear Cache
    </button>
  </header>

  <div class="scroll-area">

    <!-- ════════════════════════════════════════════════════════
         SECTION 1: Overview KPIs
         ════════════════════════════════════════════════════════ -->
    <section class="section">
      <div class="section-header">
        <div class="section-icon">📊</div>
        <div>
          <h2 class="section-title">Overview</h2>
          <p class="section-desc">Key performance indicators at a glance</p>
        </div>
      </div>
      <div class="kpi-row">
        <div class="kpi" style="--accent: #818cf8;">
          <span class="kpi-label">Hit Rate</span>
          <span class="kpi-val">{stats ? (stats.aggregate.hit_rate * 100).toFixed(1) : '—'}%</span>
          <span class="kpi-sub">{stats?.aggregate.total_hits ?? 0} / {stats?.aggregate.total_queries ?? 0} queries</span>
        </div>
        <div class="kpi" style="--accent: #34d399;">
          <span class="kpi-label">Avg Latency</span>
          <span class="kpi-val">{stats ? stats.aggregate.avg_latency_ms.toFixed(1) : '—'}<small>ms</small></span>
          <span class="kpi-sub">Hit avg {stats ? stats.aggregate.avg_hit_latency_ms.toFixed(1) : '—'}ms</span>
        </div>
        <div class="kpi" style="--accent: #a78bfa;">
          <span class="kpi-label">Cache Size</span>
          <span class="kpi-val">{stats?.aggregate.cache_size ?? 0}</span>
          <span class="kpi-sub">of {stats?.aggregate.cache_capacity?.toLocaleString() ?? '—'} max</span>
        </div>
        <div class="kpi" style="--accent: #fbbf24;">
          <span class="kpi-label">LLM Calls Saved</span>
          <span class="kpi-val">{stats ? (stats.aggregate.hit_rate * 100).toFixed(0) : '—'}%</span>
          <span class="kpi-sub">{stats?.aggregate.total_hits ?? 0} cache hits</span>
        </div>
        <div class="kpi" style="--accent: #fb7185;">
          <span class="kpi-label">Precision</span>
          <span class="kpi-val">{stats ? (stats.aggregate.precision * 100).toFixed(1) : '—'}%</span>
          <span class="kpi-sub">FHR: {stats ? (stats.aggregate.false_hit_rate * 100).toFixed(2) : '—'}%</span>
        </div>
        <div class="kpi" style="--accent: #22d3ee;">
          <span class="kpi-label">Tier 1 Rate</span>
          <span class="kpi-val">{stats && stats.aggregate.total_hits > 0 ? (stats.aggregate.tier1_rate * 100).toFixed(0) : '—'}%</span>
          <span class="kpi-sub">Exact-hash hits</span>
        </div>
      </div>
    </section>

    <!-- ════════════════════════════════════════════════════════
         SECTION 2: Cache Performance
         ════════════════════════════════════════════════════════ -->
    <section class="section">
      <div class="section-header">
        <div class="section-icon">📈</div>
        <div>
          <h2 class="section-title">Cache Performance</h2>
          <p class="section-desc">Hit rate trends, query latency, and tier distribution over time</p>
        </div>
      </div>
      <div class="chart-row-3">
        <div class="card span-2">
          <h3 class="card-title">Hit Rate Over Time</h3>
          <p class="card-hint">Percentage of queries served from cache</p>
          <div class="chart-box"><canvas id="hr"></canvas></div>
        </div>
        <div class="card">
          <h3 class="card-title">Query Breakdown</h3>
          <p class="card-hint">Tier 1 (exact) vs Tier 2 (semantic) vs miss</p>
          <div class="chart-box"><canvas id="ti"></canvas></div>
        </div>
      </div>
    </section>

    <!-- ════════════════════════════════════════════════════════
         SECTION 3: Latency Analysis
         ════════════════════════════════════════════════════════ -->
    <section class="section">
      <div class="section-header">
        <div class="section-icon">⚡</div>
        <div>
          <h2 class="section-title">Latency Analysis</h2>
          <p class="section-desc">Response time distribution and percentiles</p>
        </div>
      </div>
      <div class="chart-row-2">
        <div class="card">
          <h3 class="card-title">Query Latency</h3>
          <p class="card-hint">Per-query response time in milliseconds</p>
          <div class="chart-box"><canvas id="lt"></canvas></div>
        </div>
        <div class="card">
          <h3 class="card-title">Latency Percentiles</h3>
          <p class="card-hint">Tail latency distribution</p>
          <div class="pct-grid-inner">
            {#each [
              { key: 'p50', label: 'P50 (Median)', color: '#34d399' },
              { key: 'p90', label: 'P90', color: '#fbbf24' },
              { key: 'p95', label: 'P95', color: '#f97316' },
              { key: 'p99', label: 'P99', color: '#f87171' },
            ] as p}
              <div class="pct-block">
                <div class="pct-bar-track">
                  <div class="pct-bar-fill" style="width: {Math.min(((stats?.latency_distribution?.[p.key as keyof typeof stats.latency_distribution] ?? 0) / Math.max(stats?.latency_distribution?.p99 ?? 1, 1)) * 100, 100)}%; background: {p.color};"></div>
                </div>
                <div class="pct-meta">
                  <span class="pct-name">{p.label}</span>
                  <span class="pct-ms" style="color: {p.color};">{stats?.latency_distribution?.[p.key as keyof typeof stats.latency_distribution] ?? 0}<small>ms</small></span>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </section>

    <!-- ════════════════════════════════════════════════════════
         SECTION 4: Threshold Calibration & Bandit
         ════════════════════════════════════════════════════════ -->
    <section class="section">
      <div class="section-header">
        <div class="section-icon">🎯</div>
        <div>
          <h2 class="section-title">Threshold Calibration</h2>
          <p class="section-desc">Conformal thresholds, Thompson Sampling arm selection, and similarity scores</p>
        </div>
      </div>
      <div class="chart-row-3">
        <div class="card">
          <h3 class="card-title">Thompson Sampling Arms</h3>
          <p class="card-hint">Expected reward per threshold arm{stats?.bandit ? ` · Best: τ=${stats.bandit.best_threshold}` : ''}</p>
          <div class="chart-box"><canvas id="ba"></canvas></div>
        </div>
        <div class="card">
          <h3 class="card-title">Similarity Distribution</h3>
          <p class="card-hint">Cosine similarity scores of cache hits</p>
          <div class="chart-box"><canvas id="sm"></canvas></div>
        </div>
        <div class="card">
          <h3 class="card-title">Conformal Threshold Trace</h3>
          <p class="card-hint">Per-entry τ values over time</p>
          <div class="chart-box"><canvas id="th"></canvas></div>
        </div>
      </div>
    </section>

    <!-- ════════════════════════════════════════════════════════
         SECTION 5: System Components
         ════════════════════════════════════════════════════════ -->
    <section class="section">
      <div class="section-header">
        <div class="section-icon">⚙️</div>
        <div>
          <h2 class="section-title">System Components</h2>
          <p class="section-desc">Internals of cache admission, eviction, bandit, and providers</p>
        </div>
      </div>
      <div class="detail-grid">
        <div class="detail-card">
          <div class="detail-header">
            <span class="detail-badge evict">Eviction</span>
            <span class="detail-algo">CostGDSF</span>
          </div>
          <div class="row"><span>Total Evictions</span><span class="mono">{stats?.eviction?.total_evictions ?? 0}</span></div>
          <div class="row"><span>Tracked Entries</span><span class="mono">{stats?.eviction?.tracked_entries ?? 0}</span></div>
          <div class="row"><span>Inflation L</span><span class="mono">{stats?.eviction?.inflation_factor ?? 0}</span></div>
          <div class="row"><span>Avg Priority</span><span class="mono">{stats?.eviction?.avg_priority ?? 0}</span></div>
        </div>
        <div class="detail-card">
          <div class="detail-header">
            <span class="detail-badge admit">Admission</span>
            <span class="detail-algo">W-TinyLFU</span>
          </div>
          <div class="row"><span>Checks</span><span class="mono">{stats?.admission?.total_checks ?? 0}</span></div>
          <div class="row"><span>Admitted</span><span class="mono">{stats?.admission?.total_admitted ?? 0}</span></div>
          <div class="row"><span>Rejected</span><span class="mono">{stats?.admission?.total_rejected ?? 0}</span></div>
          <div class="row"><span>Rate</span><span class="mono">{stats?.admission?.admission_rate ? (stats.admission.admission_rate * 100).toFixed(1) + '%' : '—'}</span></div>
        </div>
        <div class="detail-card">
          <div class="detail-header">
            <span class="detail-badge bandit">Bandit</span>
            <span class="detail-algo">Thompson</span>
          </div>
          <div class="row"><span>Best Threshold</span><span class="mono">{stats?.bandit?.best_threshold ?? '—'}</span></div>
          <div class="row"><span>Total Updates</span><span class="mono">{stats?.bandit?.total_updates ?? 0}</span></div>
          <div class="row"><span>Drift Resets</span><span class="mono">{stats?.bandit?.drift_resets ?? 0}</span></div>
          <div class="row"><span>Shard</span><span class="mono">{stats?.bandit?.shard_id ?? '—'}</span></div>
        </div>
        <div class="detail-card">
          <div class="detail-header">
            <span class="detail-badge threshold">Thresholds</span>
            <span class="detail-algo">Conformal</span>
          </div>
          <div class="row"><span>Entries</span><span class="mono">{stats?.thresholds?.total_entries ?? 0}</span></div>
          <div class="row"><span>Calibrated</span><span class="mono">{stats?.thresholds?.calibrated_entries ?? 0}</span></div>
          <div class="row"><span>Target ε</span><span class="mono">{stats?.thresholds?.target_error_rate ?? '—'}</span></div>
          <div class="row"><span>Default τ</span><span class="mono">{stats?.thresholds?.default_threshold ?? '—'}</span></div>
        </div>
        <div class="detail-card">
          <div class="detail-header">
            <span class="detail-badge embed">Embedding</span>
            <span class="detail-algo">SentenceTransformer</span>
          </div>
          <div class="row"><span>Encode Calls</span><span class="mono">{stats?.embedding_service?.encode_calls ?? 0}</span></div>
          <div class="row"><span>Cache Hits</span><span class="mono">{stats?.embedding_service?.cache_hits ?? 0}</span></div>
          <div class="row"><span>Avg Encode</span><span class="mono">{stats?.embedding_service?.avg_encode_ms ?? 0}ms</span></div>
        </div>
        <div class="detail-card">
          <div class="detail-header">
            <span class="detail-badge llm">LLM</span>
            <span class="detail-algo">{stats?.llm?.backend ?? '—'}/{stats?.llm?.model ?? '—'}</span>
          </div>
          <div class="row"><span>Total Calls</span><span class="mono">{stats?.llm?.total_calls ?? 0}</span></div>
          <div class="row"><span>Avg Latency</span><span class="mono">{stats?.llm?.avg_latency_ms ?? 0}ms</span></div>
          <div class="row"><span>Circuit State</span><span class="mono upper">{stats?.llm?.circuit_state ?? '—'}</span></div>
        </div>
      </div>
    </section>

  </div>
</div>

<style>
  .analytics {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #0c0c14;
  }

  /* ── Topbar ──────────────────────────────────────────── */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    border-bottom: 1px solid rgba(99,102,241,0.08);
    flex-shrink: 0;
  }
  .page-title { font-size: 17px; font-weight: 700; color: #e2e8f0; letter-spacing: -0.3px; }
  .btn-clear {
    display: flex; align-items: center; gap: 6px;
    padding: 7px 14px; font-size: 12px; font-weight: 500; font-family: 'Inter', sans-serif;
    color: #94a3b8; background: rgba(148,163,184,0.06); border: 1px solid rgba(148,163,184,0.1);
    border-radius: 8px; cursor: pointer; transition: all 0.15s;
  }
  .btn-clear:hover { color: #f87171; border-color: rgba(248,113,113,0.25); background: rgba(248,113,113,0.06); }

  .scroll-area {
    flex: 1; overflow-y: auto; padding: 0 24px 40px;
  }
  .scroll-area::-webkit-scrollbar { width: 6px; }
  .scroll-area::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.12); border-radius: 3px; }

  /* ── Section ─────────────────────────────────────────── */
  .section {
    padding-top: 24px;
    padding-bottom: 8px;
  }
  .section + .section {
    border-top: 1px solid rgba(99,102,241,0.06);
  }
  .section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }
  .section-icon {
    font-size: 22px;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(99,102,241,0.06);
    border-radius: 10px;
    flex-shrink: 0;
  }
  .section-title {
    font-size: 15px;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.2px;
    line-height: 1.2;
  }
  .section-desc {
    font-size: 12px;
    color: #475569;
    margin-top: 1px;
  }

  /* ── KPI ──────────────────────────────────────────────── */
  .kpi-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }
  .kpi {
    background: rgba(15,15,26,0.5);
    border: 1px solid rgba(99,102,241,0.07);
    border-radius: 12px; padding: 14px 12px; text-align: center;
    transition: border-color 0.15s;
  }
  .kpi:hover { border-color: rgba(99,102,241,0.18); }
  .kpi-label { display: block; font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.7px; color: #475569; font-weight: 600; margin-bottom: 4px; }
  .kpi-val { display: block; font-size: 22px; font-weight: 800; color: var(--accent); font-family: 'JetBrains Mono', monospace; letter-spacing: -0.5px; }
  .kpi-val small { font-size: 12px; font-weight: 500; opacity: 0.5; }
  .kpi-sub { display: block; font-size: 10px; color: #374151; margin-top: 3px; }

  /* ── Chart Cards ─────────────────────────────────────── */
  .chart-row-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .chart-row-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
  .card {
    background: rgba(15,15,26,0.5);
    border: 1px solid rgba(99,102,241,0.07);
    border-radius: 12px; padding: 16px;
  }
  .card.span-2 { grid-column: span 2; }
  .card-title { font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 2px; }
  .card-hint { font-size: 11px; color: #334155; margin-bottom: 10px; }
  .chart-box { height: 190px; position: relative; }

  /* ── Percentile bars ─────────────────────────────────── */
  .pct-grid-inner {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding-top: 8px;
  }
  .pct-block {}
  .pct-bar-track {
    width: 100%;
    height: 6px;
    background: rgba(99,102,241,0.06);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 5px;
  }
  .pct-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
  }
  .pct-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .pct-name { font-size: 11px; color: #64748b; }
  .pct-ms { font-size: 14px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .pct-ms small { font-size: 10px; font-weight: 400; opacity: 0.5; }

  /* ── Detail Grid ─────────────────────────────────────── */
  .detail-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .detail-card {
    background: rgba(15,15,26,0.5);
    border: 1px solid rgba(99,102,241,0.07);
    border-radius: 12px; padding: 14px;
  }
  .detail-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  .detail-badge {
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 5px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .detail-badge.evict { background: rgba(239,68,68,0.1); color: #f87171; }
  .detail-badge.admit { background: rgba(16,185,129,0.1); color: #34d399; }
  .detail-badge.bandit { background: rgba(99,102,241,0.1); color: #818cf8; }
  .detail-badge.threshold { background: rgba(245,158,11,0.1); color: #fbbf24; }
  .detail-badge.embed { background: rgba(139,92,246,0.1); color: #a78bfa; }
  .detail-badge.llm { background: rgba(34,211,238,0.1); color: #22d3ee; }
  .detail-algo { font-size: 10px; color: #374151; font-family: 'JetBrains Mono', monospace; }

  .row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid rgba(99,102,241,0.04); font-size: 12px;
  }
  .row:last-child { border-bottom: none; }
  .row span:first-child { color: #475569; }
  .mono { font-family: 'JetBrains Mono', monospace; color: #94a3b8; font-size: 11px; }
  .upper { text-transform: uppercase; }

  /* ── Responsive ──────────────────────────────────────── */
  @media (max-width: 1100px) {
    .kpi-row { grid-template-columns: repeat(3, 1fr); }
    .chart-row-3 { grid-template-columns: repeat(2, 1fr); }
    .chart-row-2 { grid-template-columns: 1fr; }
    .card.span-2 { grid-column: span 1; }
    .detail-grid { grid-template-columns: repeat(2, 1fr); }
  }
  @media (max-width: 700px) {
    .kpi-row { grid-template-columns: repeat(2, 1fr); }
    .chart-row-3, .chart-row-2, .detail-grid { grid-template-columns: 1fr; }
  }
</style>
