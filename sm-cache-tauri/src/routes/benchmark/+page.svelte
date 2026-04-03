<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    startBenchmark,
    getBenchmarkStatus,
    listBenchmarkResults,
    getBenchmarkResult,
    type BenchmarkProgress,
    type BenchmarkResult
  } from '$lib/api';
  import Chart from 'chart.js/auto';

  // State
  let numQuestions = $state(500);
  let cacheCapacity = $state(200);
  let selectedDataset = $state('nq');
  let isRunning = $state(false);
  let progress = $state<BenchmarkProgress | null>(null);
  let pollInterval: ReturnType<typeof setInterval>;

  let resultsHistory = $state<any[]>([]);
  let currentResult = $state<BenchmarkResult | null>(null);
  let selectedRunId = $state<string | null>(null);
  let historyFilter = $state<string>('all');

  // Chart instances
  let hitRateChart: Chart | null = null;
  let latencyChart: Chart | null = null;
  let hitRateCanvas = $state<HTMLCanvasElement>();
  let latencyCanvas = $state<HTMLCanvasElement>();

  onMount(async () => {
    await fetchHistory();
  });

  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
    if (hitRateChart) hitRateChart.destroy();
    if (latencyChart) latencyChart.destroy();
  });

  async function fetchHistory() {
    try {
      const res = await listBenchmarkResults();
      resultsHistory = res.results;
      if (resultsHistory.length > 0 && !currentResult) {
        await selectResult(resultsHistory[0].run_id);
      }
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  }

  async function selectResult(runId: string) {
    try {
      selectedRunId = runId;
      currentResult = await getBenchmarkResult(runId);
      renderCharts();
    } catch (err) {
      console.error("Failed to load result:", err);
    }
  }

  async function handleStart() {
    if (isRunning) return;
    
    try {
      const res = await startBenchmark({
        num_questions: numQuestions,
        capacity: cacheCapacity,
        dataset: selectedDataset,
      });

      if (res.run_id) {
        isRunning = true;
        currentResult = null;
        selectedRunId = null;
        
        // Start polling
        pollInterval = setInterval(async () => {
          try {
            const p = await getBenchmarkStatus(res.run_id!);
            progress = p;
            
            if (p.status === 'done' || p.status === 'error') {
              clearInterval(pollInterval);
              isRunning = false;
              if (p.status === 'done') {
                await fetchHistory();
                await selectResult(p.run_id);
              }
            }
          } catch (e) {
            console.error("Poll error", e);
          }
        }, 1000);
      }
    } catch (err) {
      console.error("Failed to start benchmark:", err);
      alert("Failed to start benchmark. Make sure backend is running.");
    }
  }

  // Define colors for strategies to keep charts consistent
  const strategyColors: Record<string, string> = {
    'ContextualCache (Ours)': '#3b82f6', // Bright blue
    'GPTCache': '#f59e0b',               // Amber
    'MeanCache': '#10b981',              // Emerald
    'vCache': '#8b5cf6',                 // Violet
    'RAGCache': '#ec4899',               // Pink
    'No-Admission': '#64748b',           // Slate
    'Exact-Match LRU': '#ef4444',        // Red
  };

  function getColor(name: string): string {
    return strategyColors[name] || '#94a3b8';
  }

  function renderCharts() {
    if (!currentResult || !hitRateCanvas || !latencyCanvas) return;

    // Destroy existing
    if (hitRateChart) hitRateChart.destroy();
    if (latencyChart) latencyChart.destroy();

    // Sort strategies logically: Ours first, then baselines
    const strategies = [...currentResult.strategies].sort((a, b) => {
      if (a.name.includes('(Ours)')) return -1;
      if (b.name.includes('(Ours)')) return 1;
      // Sort by hit rate descending for the rest
      return b.hit_rate - a.hit_rate;
    });

    const labels = strategies.map(s => s.name.replace(' (Ours)', ''));
    const colors = strategies.map(s => getColor(s.name));

    // Hit Rate & Precision Chart
    hitRateChart = new Chart(hitRateCanvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Hit Rate (%)',
            data: strategies.map(s => s.hit_rate * 100),
            backgroundColor: colors,
            borderWidth: 0,
            borderRadius: 4
          },
          {
            label: 'Precision (%)',
            data: strategies.map(s => s.precision * 100),
            backgroundColor: colors.map(c => c + '80'), // 50% opacity
            borderWidth: 0,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8' } },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${(ctx.raw as number)?.toFixed(1) || 0}%`
            }
          }
        },
        scales: {
          y: { 
            beginAtZero: true, 
            max: 100,
            grid: { color: '#1e1e2e' },
            ticks: { color: '#64748b' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });

    // Latency Chart
    latencyChart = new Chart(latencyCanvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Avg Hit Latency (ms)',
            data: strategies.map(s => s.avg_hit_latency_ms),
            backgroundColor: '#34d399',
            borderRadius: 4
          },
          {
            label: 'Avg Overall Latency (ms)',
            data: strategies.map(s => s.avg_overall_latency_ms),
            backgroundColor: '#6366f1',
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
          y: {
            type: 'logarithmic',
            grid: { color: '#1e1e2e' },
            ticks: { color: '#64748b' },
            title: { display: true, text: 'Latency (ms) - Log Scale', color: '#64748b' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }

  // Keep charts rendered if data exists and canvas mounts
  $effect(() => {
    if (currentResult && hitRateCanvas && latencyCanvas) {
      renderCharts();
    }
  });

  function formatDate(ts: number) {
    return new Date(ts * 1000).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
  }

  function datasetLabel(ds?: string): string {
    if (ds === 'squad') return 'SQuAD';
    return 'Natural Questions';
  }

  function filteredHistory(): any[] {
    if (historyFilter === 'all') return resultsHistory;
    return resultsHistory.filter((r: any) => (r.dataset || 'nq') === historyFilter);
  }
</script>

<div class="benchmark-container">
  <!-- Header -->
  <header class="page-header">
    <div class="title-section">
      <h1>Cache Benchmark Suite</h1>
      <p class="subtitle">Evaluate ContextualCache against 6 research baselines on public QA datasets</p>
    </div>
  </header>

  <div class="content-grid">
    <!-- Left Column: Controls & History -->
    <div class="sidebar-col">
      <!-- Run Controls -->
      <div class="card run-controls">
        <h3>New Benchmark Run</h3>
        
        <div class="input-group">
          <label for="numQs">Number of Questions</label>
          <div class="slider-wrapper">
            <input type="range" id="numQs" min="10" max="1000" step="10" bind:value={numQuestions} disabled={isRunning} />
            <span class="val-display">{numQuestions}</span>
          </div>
          <small>Includes {Math.round(numQuestions * 0.3)} semantic paraphrase variants</small>
        </div>

        <div class="input-group">
          <label for="capacity">Cache Capacity (Entries)</label>
          <div class="slider-wrapper">
            <input type="range" id="capacity" min="50" max="1000" step="50" bind:value={cacheCapacity} disabled={isRunning} />
            <span class="val-display">{cacheCapacity}</span>
          </div>
        </div>

        <div class="input-group">
          <label for="dataset">Dataset</label>
          <select id="dataset" class="select-input" bind:value={selectedDataset} disabled={isRunning}>
            <option value="nq">Natural Questions (NQ)</option>
            <option value="squad">SQuAD</option>
          </select>
        </div>

        <button 
          class="btn-primary" 
          onclick={handleStart} 
          disabled={isRunning}
        >
          {isRunning ? 'Running Benchmark...' : 'Start Benchmark'}
        </button>

        {#if isRunning && progress}
          <div class="progress-box">
            <div class="status-text">
              <span class="spinner"></span>
              {#if progress.status === 'running'}
                Evaluating <strong>{progress.current_strategy}</strong>
              {:else if progress.status === 'error'}
                <span style="color: #f87171">Error: {progress.error}</span>
              {:else}
                Starting...
              {/if}
            </div>
            
            <div class="progress-stats">
              <div class="stat-mini">
                <span class="label">Strategy</span>
                <span class="val">{progress.completed_strategies} / {progress.total_strategies}</span>
              </div>
              <div class="stat-mini">
                <span class="label">Queries (Current)</span>
                <span class="val">{progress.completed_queries} / {progress.total_queries}</span>
              </div>
              <div class="stat-mini">
                <span class="label">Time</span>
                <span class="val">{Number(progress.elapsed_s).toFixed(1)}s</span>
              </div>
            </div>

            <div class="progress-bar-bg">
              <div 
                class="progress-bar-fill" 
                style="width: {progress.total_strategies > 0 ? (progress.completed_strategies / progress.total_strategies) * 100 : 0}%"
              ></div>
            </div>
          </div>
        {/if}
      </div>

      <!-- History -->
      <div class="card history-list">
        <h3>Benchmark History</h3>
        <div class="dataset-tabs">
          <button class="tab-btn {historyFilter === 'all' ? 'active' : ''}" onclick={() => historyFilter = 'all'}>All</button>
          <button class="tab-btn {historyFilter === 'nq' ? 'active' : ''}" onclick={() => historyFilter = 'nq'}>NQ</button>
          <button class="tab-btn {historyFilter === 'squad' ? 'active' : ''}" onclick={() => historyFilter = 'squad'}>SQuAD</button>
        </div>
        {#if filteredHistory().length === 0}
          <div class="empty-state">No past runs found.</div>
        {:else}
          <div class="run-items">
            {#each filteredHistory() as run}
              <button
                class="history-item {selectedRunId === run.run_id ? 'active' : ''}"
                onclick={() => selectResult(run.run_id)}
                disabled={isRunning}
              >
                <div class="history-main">
                  <span class="id">#{run.run_id}</span>
                  <span class="date">{formatDate(run.timestamp)}</span>
                </div>
                <div class="history-sub">
                  <span class="dataset-badge">{(run.dataset || 'nq').toUpperCase()}</span>
                  <span>{run.num_questions} Qs</span>
                  <span>Cap: {run.cache_capacity}</span>
                </div>
              </button>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- Right Column: Results -->
    <div class="main-col">
      {#if currentResult}
        <div class="results-header">
          <h2>Results: Run #{currentResult.run_id}</h2>
          <div class="run-meta">
            <span class="meta-pill">Dataset: {datasetLabel(currentResult.dataset)} ({currentResult.num_questions} unique, {currentResult.num_paraphrases} paraphrases)</span>
            <span class="meta-pill">Capacity: {currentResult.cache_capacity} entries</span>
          </div>
        </div>

        <div class="charts-row">
          <div class="card chart-card">
            <h3>Hit Rate & Precision</h3>
            <div class="canvas-container">
              <canvas bind:this={hitRateCanvas}></canvas>
            </div>
          </div>
          <div class="card chart-card">
            <h3>Latency Analysis</h3>
            <div class="canvas-container">
              <canvas bind:this={latencyCanvas}></canvas>
            </div>
          </div>
        </div>

        <div class="card table-card">
          <h3>Detailed Metrics Comparison</h3>
          <div class="table-responsive">
            <table>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Hit Rate</th>
                  <th>Precision</th>
                  <th>F1 Score</th>
                  <th>Hit Latency</th>
                  <th>Overall Latency</th>
                  <th>LLM Calls (Misses)</th>
                  <th>Time Taken</th>
                </tr>
              </thead>
              <tbody>
                {#each [...currentResult.strategies].sort((a,b) => b.hit_rate - a.hit_rate) as s}
                  <tr class={s.name.includes('(Ours)') ? 'highlight-row' : ''}>
                    <td class="strategy-cell">
                      <span class="color-dot" style="background-color: {getColor(s.name)}"></span>
                      <strong>{s.name}</strong>
                    </td>
                    <td class="metric">{(s.hit_rate * 100).toFixed(1)}%</td>
                    <td class="metric">{(s.precision * 100).toFixed(1)}%</td>
                    <td class="metric">{s.f1_score.toFixed(3)}</td>
                    <td class="metric latency-col">{s.avg_hit_latency_ms.toFixed(1)}ms</td>
                    <td class="metric latency-col">{s.avg_overall_latency_ms.toFixed(1)}ms</td>
                    <td class="metric call-col">{s.total_misses}</td>
                    <td class="metric">{s.total_time_s.toFixed(1)}s</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      {:else if !isRunning}
        <div class="results-placeholder">
          <div class="placeholder-icon">📊</div>
          <h3>No Results Selected</h3>
          <p>Start a new benchmark run or select one from history to view comparisons.</p>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .benchmark-container {
    padding: 30px 40px;
    max-width: 1400px;
    margin: 0 auto;
    color: #e2e8f0;
  }

  .page-header {
    margin-bottom: 30px;
  }

  h1 {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 6px;
    background: linear-gradient(135deg, #fff, #a5b4fc);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .subtitle {
    color: #94a3b8;
    font-size: 15px;
  }

  .content-grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 24px;
    align-items: start;
  }

  .card {
    background: #151522;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  }

  .card h3 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 20px;
    color: #e0e7ff;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }

  .sidebar-col {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* Controls */
  .input-group {
    margin-bottom: 20px;
  }

  .input-group label {
    display: block;
    font-size: 13px;
    color: #cbd5e1;
    margin-bottom: 8px;
  }

  .slider-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  input[type="range"] {
    flex: 1;
    accent-color: #6366f1;
  }

  .val-display {
    font-family: monospace;
    font-size: 14px;
    background: #1e1e2e;
    padding: 4px 10px;
    border-radius: 6px;
    min-width: 50px;
    text-align: center;
  }

  .input-group small {
    display: block;
    margin-top: 6px;
    color: #64748b;
    font-size: 12px;
  }

  .btn-primary {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border: none;
    padding: 12px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
  }

  .btn-primary:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
  }

  .btn-primary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    background: #475569;
    box-shadow: none;
  }

  /* Progress Box */
  .progress-box {
    margin-top: 24px;
    background: #0f0f1a;
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 16px;
  }

  .status-text {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
    color: #e2e8f0;
    margin-bottom: 16px;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(99, 102, 241, 0.3);
    border-top-color: #8b5cf6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .progress-stats {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-bottom: 16px;
  }

  .stat-mini {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .stat-mini .label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
  }

  .stat-mini .val {
    font-size: 13px;
    font-family: monospace;
    color: #a5b4fc;
  }

  .progress-bar-bg {
    height: 6px;
    background: #1e1e2e;
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    transition: width 0.3s ease;
  }

  /* History */
  .history-list {
    max-height: 400px;
    overflow-y: auto;
  }

  .run-items {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .history-item {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 12px;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s;
    color: #cbd5e1;
  }

  .history-item:hover:not(:disabled) {
    background: rgba(255,255,255,0.02);
    border-color: rgba(255,255,255,0.1);
  }

  .history-item.active {
    background: rgba(99, 102, 241, 0.1);
    border-color: rgba(99, 102, 241, 0.3);
    color: #fff;
  }

  .history-main {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
  }

  .history-main .id {
    font-family: monospace;
    font-weight: 600;
    color: #a5b4fc;
  }

  .history-main .date {
    font-size: 12px;
    color: #64748b;
  }

  .history-sub {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #94a3b8;
  }

  /* Results Area */
  .main-col {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .results-header h2 {
    font-size: 22px;
    margin-bottom: 12px;
  }

  .run-meta {
    display: flex;
    gap: 12px;
  }

  .meta-pill {
    background: #1e1e2e;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 13px;
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.2);
  }

  .charts-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }

  .canvas-container {
    height: 300px;
    width: 100%;
    position: relative;
  }

  /* Table */
  .table-responsive {
    overflow-x: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }

  th {
    text-align: left;
    padding: 12px 16px;
    color: #94a3b8;
    font-weight: 500;
    font-size: 13px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }

  td {
    padding: 14px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }

  .highlight-row {
    background: rgba(99, 102, 241, 0.08);
  }

  .strategy-cell {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .color-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }

  .metric {
    font-family: monospace;
    text-align: right;
  }

  th:not(:first-child) {
    text-align: right;
  }

  .latency-col { color: #34d399; }
  .call-col { color: #f87171; }

  /* Empty state */
  .results-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 500px;
    background: #151522;
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 16px;
    text-align: center;
    color: #64748b;
  }

  .placeholder-icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.5;
  }

  .results-placeholder h3 {
    font-size: 20px;
    color: #e2e8f0;
    margin-bottom: 8px;
  }

  /* Dataset selector */
  .select-input {
    width: 100%;
    background: #1e1e2e;
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 14px;
    cursor: pointer;
    appearance: auto;
  }

  .select-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Dataset filter tabs */
  .dataset-tabs {
    display: flex;
    gap: 6px;
    margin-bottom: 16px;
  }

  .tab-btn {
    flex: 1;
    background: transparent;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 600;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s;
  }

  .tab-btn:hover {
    background: rgba(255,255,255,0.03);
  }

  .tab-btn.active {
    background: rgba(99, 102, 241, 0.15);
    border-color: rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
  }

  .dataset-badge {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
</style>
