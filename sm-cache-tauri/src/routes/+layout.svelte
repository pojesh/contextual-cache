<script lang="ts">
  import { page } from '$app/stores';
  import { getHealth } from '$lib/api';
  import { onMount, onDestroy } from 'svelte';

  let { children } = $props();
  let connected = $state(false);
  let healthInterval: ReturnType<typeof setInterval>;

  async function checkHealth() {
    try {
      await getHealth();
      connected = true;
    } catch {
      connected = false;
    }
  }

  onMount(() => {
    checkHealth();
    healthInterval = setInterval(checkHealth, 5000);
  });

  onDestroy(() => clearInterval(healthInterval));

  const navItems = [
    { path: '/', label: 'Chat', icon: 'M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z' },
    { path: '/compare', label: 'Compare', icon: 'M9 3v18m6-18v18M3 9h18M3 15h18' },
    { path: '/analytics', label: 'Analytics', icon: 'M18 20V10M12 20V4M6 20v-6' },
    { path: '/benchmark', label: 'Benchmark', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  ];
</script>

<div class="shell">
  <!-- Sidebar -->
  <nav class="sidebar">
    <div class="sidebar-top">
      <div class="brand">
        <div class="brand-icon">⚡</div>
        <span class="brand-text">ContextualCache</span>
      </div>
      <div class="nav-links">
        {#each navItems as item}
          <a
            href={item.path}
            class="nav-link"
            class:active={$page.url.pathname === item.path}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d={item.icon}/>
            </svg>
            {item.label}
          </a>
        {/each}
      </div>
    </div>
    <div class="sidebar-bottom">
      <div class="status-pill" class:online={connected} class:offline={!connected}>
        <span class="dot"></span>
        {connected ? 'Backend Online' : 'Disconnected'}
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <main class="main-content">
    {@render children()}
  </main>
</div>

<style>
  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(body) {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0c0c14;
    color: #e2e8f0;
    overflow: hidden;
    height: 100vh;
  }
  :global(a) { text-decoration: none; color: inherit; }

  .shell {
    display: flex;
    height: 100vh;
    width: 100vw;
  }

  /* ── Sidebar ────────────────────────────────────────────── */
  .sidebar {
    width: 220px;
    min-width: 220px;
    background: #0f0f1a;
    border-right: 1px solid rgba(99, 102, 241, 0.1);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 20px 12px;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 8px;
    margin-bottom: 28px;
  }

  .brand-icon {
    font-size: 22px;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.25);
  }

  .brand-text {
    font-size: 15px;
    font-weight: 700;
    background: linear-gradient(135deg, #c7d2fe, #e0e7ff);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.3px;
  }

  .nav-links {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .nav-link {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
    color: #64748b;
    transition: all 0.15s ease;
  }

  .nav-link:hover {
    color: #c7d2fe;
    background: rgba(99, 102, 241, 0.06);
  }

  .nav-link.active {
    color: #e0e7ff;
    background: rgba(99, 102, 241, 0.12);
    box-shadow: inset 3px 0 0 #6366f1;
  }

  .sidebar-bottom {
    padding: 0 4px;
  }

  .status-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 11px;
    font-weight: 500;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid;
  }

  .status-pill.online {
    color: #34d399;
    border-color: rgba(52, 211, 153, 0.2);
    background: rgba(52, 211, 153, 0.06);
  }

  .status-pill.offline {
    color: #f87171;
    border-color: rgba(248, 113, 113, 0.2);
    background: rgba(248, 113, 113, 0.06);
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }

  .online .dot {
    background: #34d399;
    box-shadow: 0 0 6px #34d399;
    animation: blink 2s infinite;
  }

  .offline .dot { background: #f87171; }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
  }

  /* ── Main Content ───────────────────────────────────────── */
  .main-content {
    flex: 1;
    overflow-y: auto;
    height: 100vh;
    background: #0c0c14;
  }
</style>
