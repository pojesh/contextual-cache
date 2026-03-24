<script module lang="ts">
  export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    text: string;
    meta?: {
      hit: boolean;
      tier: number;
      latency_ms: number;
      similarity: number;
      entry_id?: string;
    };
    timestamp: number;
  }

  // Preserved state across page navigations
  let messages = $state<ChatMessage[]>([]);
  let input = $state('');
  let sessionId = $state('session-' + Math.random().toString(36).slice(2, 8));
</script>

<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { sendQuery, getStats } from '$lib/api';
  import type { QueryResponse } from '$lib/api';

  // ── Local UI State ─────────────────────────────────────────
  let sending = $state(false);
  let cacheStats = $state({ hit_rate: 0, total_queries: 0, total_hits: 0, cache_size: 0, hit_latency: 0 });

  let chatContainer: HTMLElement;

  // ── Scroll helper ──────────────────────────────────────────
  async function scrollToBottom() {
    await tick();
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  // ── Send query ─────────────────────────────────────────────
  async function handleSend() {
    const text = input.trim();
    if (!text || sending) return;

    // Add user message
    messages = [...messages, {
      id: crypto.randomUUID(),
      role: 'user',
      text,
      timestamp: Date.now(),
    }];
    input = '';
    sending = true;
    await scrollToBottom();

    try {
      const res: QueryResponse = await sendQuery({
        query: text,
        session_id: sessionId,
      });

      messages = [...messages, {
        id: crypto.randomUUID(),
        role: 'assistant',
        text: res.response,
        meta: {
          hit: res.hit,
          tier: res.tier,
          latency_ms: res.latency_ms,
          similarity: res.similarity ?? 0,
          entry_id: res.entry_id ?? undefined,
        },
        timestamp: Date.now(),
      }];

      // Refresh stats
      try {
        const s = await getStats();
        cacheStats = {
          hit_rate: s.aggregate.hit_rate,
          total_queries: s.aggregate.total_queries,
          total_hits: s.aggregate.total_hits,
          cache_size: s.aggregate.cache_size,
          hit_latency: s.aggregate.avg_hit_latency_ms,
        };
      } catch { /* ignore */ }
    } catch (e: any) {
      messages = [...messages, {
        id: crypto.randomUUID(),
        role: 'assistant',
        text: `Error: ${e.message || 'Failed to reach backend'}`,
        timestamp: Date.now(),
      }];
    } finally {
      sending = false;
      await scrollToBottom();
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function newSession() {
    sessionId = 'session-' + Math.random().toString(36).slice(2, 8);
    messages = [];
  }

  onMount(async () => {
    scrollToBottom();
    try {
      const s = await getStats();
      cacheStats = {
        hit_rate: s.aggregate.hit_rate,
        total_queries: s.aggregate.total_queries,
        total_hits: s.aggregate.total_hits,
        cache_size: s.aggregate.cache_size,
        hit_latency: s.aggregate.avg_hit_latency_ms,
      };
    } catch { /* backend not ready yet */ }
  });
</script>

<svelte:head>
  <title>Chat — ContextualCache</title>
</svelte:head>

<div class="chat-page">
  <!-- Top Bar -->
  <header class="topbar">
    <div class="topbar-left">
      <h1 class="page-title">Chat</h1>
      <span class="session-tag">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
        {sessionId}
      </span>
    </div>
    <div class="topbar-right">
      <div class="stat-chip">
        <span class="stat-label">Hit Rate</span>
        <span class="stat-value accent-indigo">{(cacheStats.hit_rate * 100).toFixed(1)}%</span>
      </div>
      <div class="stat-chip">
        <span class="stat-label">Hit Latency</span>
        <span class="stat-value accent-emerald">{cacheStats.hit_latency.toFixed(1)}<small style="font-size:10px;font-weight:normal;">ms</small></span>
      </div>
      <div class="stat-chip">
        <span class="stat-label">Queries</span>
        <span class="stat-value">{cacheStats.total_queries}</span>
      </div>
      <div class="stat-chip">
        <span class="stat-label">Cached</span>
        <span class="stat-value">{cacheStats.cache_size}</span>
      </div>
      <button class="btn-icon" onclick={newSession} title="New Session">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>
      </button>
    </div>
  </header>

  <!-- Chat Messages -->
  <div class="chat-scroll" bind:this={chatContainer}>
    {#if messages.length === 0}
      <div class="empty-state">
        <div class="empty-icon">💬</div>
        <h2>Start a Conversation</h2>
        <p>Ask anything — responses are cached semantically so similar questions get instant answers.</p>
        <div class="hint-grid">
          <button class="hint" onclick={() => { input = 'What is machine learning?'; handleSend(); }}>What is machine learning?</button>
          <button class="hint" onclick={() => { input = 'Explain neural networks'; handleSend(); }}>Explain neural networks</button>
          <button class="hint" onclick={() => { input = 'How does gradient descent work?'; handleSend(); }}>How does gradient descent work?</button>
          <button class="hint" onclick={() => { input = 'What are transformers in AI?'; handleSend(); }}>What are transformers in AI?</button>
        </div>
      </div>
    {:else}
      <div class="messages">
        {#each messages as msg (msg.id)}
          <div class="msg-row" class:user={msg.role === 'user'} class:assistant={msg.role === 'assistant'}>
            {#if msg.role === 'user'}
              <div class="msg-bubble user-bubble">
                <p>{msg.text}</p>
              </div>
            {:else}
              <div class="msg-bubble assistant-bubble">
                {#if msg.meta}
                  <div class="meta-bar">
                    <span class="meta-badge" class:hit={msg.meta.hit} class:miss={!msg.meta.hit}>
                      {msg.meta.hit ? `Cache Hit · Tier ${msg.meta.tier}` : 'Cache Miss · LLM'}
                    </span>
                    <span class="meta-latency">{msg.meta.latency_ms.toFixed(0)}ms</span>
                    {#if msg.meta.hit && msg.meta.similarity > 0}
                      <span class="meta-sim">sim {msg.meta.similarity.toFixed(3)}</span>
                    {/if}
                  </div>
                {/if}
                <p class="response-text">{msg.text}</p>
              </div>
            {/if}
          </div>
        {/each}
        {#if sending}
          <div class="msg-row assistant">
            <div class="msg-bubble assistant-bubble typing">
              <span class="dot-one">·</span><span class="dot-two">·</span><span class="dot-three">·</span>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Input Bar -->
  <div class="input-bar">
    <input
      type="text"
      class="chat-input"
      placeholder="Type a message…"
      bind:value={input}
      onkeydown={handleKeydown}
      disabled={sending}
    />
    <button class="send-btn" onclick={handleSend} disabled={sending || !input.trim()} aria-label="Send message">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
    </button>
  </div>
</div>

<style>
  .chat-page {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #0c0c14;
  }

  /* ── Topbar ─────────────────────────────────────────────── */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    border-bottom: 1px solid rgba(99, 102, 241, 0.08);
    background: rgba(12, 12, 20, 0.9);
    backdrop-filter: blur(12px);
    flex-shrink: 0;
  }

  .topbar-left { display: flex; align-items: center; gap: 12px; }
  .topbar-right { display: flex; align-items: center; gap: 10px; }

  .page-title {
    font-size: 17px;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.3px;
  }

  .session-tag {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 8px;
    background: rgba(99, 102, 241, 0.06);
    border-radius: 6px;
    border: 1px solid rgba(99, 102, 241, 0.1);
  }

  .stat-chip {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 4px 12px;
    background: rgba(15, 15, 26, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.08);
    border-radius: 8px;
    min-width: 64px;
  }

  .stat-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #475569;
    font-weight: 600;
  }

  .stat-value {
    font-size: 14px;
    font-weight: 700;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
  }

  .stat-value.accent-indigo { color: #818cf8; }
  .stat-value.accent-emerald { color: #34d399; }

  .btn-icon {
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 8px;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
  }

  .btn-icon:hover {
    background: rgba(99, 102, 241, 0.15);
    color: #c7d2fe;
  }

  /* ── Chat Area ──────────────────────────────────────────── */
  .chat-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 24px 0;
  }

  .chat-scroll::-webkit-scrollbar { width: 6px; }
  .chat-scroll::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.15); border-radius: 3px; }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    padding: 24px;
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
    max-width: 400px;
    line-height: 1.6;
    margin-bottom: 24px;
  }

  .hint-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    max-width: 420px;
  }

  .hint {
    padding: 10px 14px;
    font-size: 12px;
    font-family: 'Inter', sans-serif;
    color: #94a3b8;
    background: rgba(99, 102, 241, 0.05);
    border: 1px solid rgba(99, 102, 241, 0.1);
    border-radius: 10px;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
  }

  .hint:hover {
    background: rgba(99, 102, 241, 0.1);
    border-color: rgba(99, 102, 241, 0.25);
    color: #c7d2fe;
  }

  /* ── Messages ───────────────────────────────────────────── */
  .messages {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 0 24px;
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
  }

  .msg-row {
    display: flex;
  }

  .msg-row.user { justify-content: flex-end; }
  .msg-row.assistant { justify-content: flex-start; }

  .msg-bubble {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 13.5px;
    line-height: 1.65;
    word-wrap: break-word;
  }

  .user-bubble {
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    color: #fff;
    border-bottom-right-radius: 4px;
  }

  .user-bubble p { margin: 0; }

  .assistant-bubble {
    background: #151520;
    color: #cbd5e1;
    border: 1px solid rgba(99, 102, 241, 0.08);
    border-bottom-left-radius: 4px;
  }

  .meta-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }

  .meta-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 5px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  .meta-badge.hit {
    background: rgba(99, 102, 241, 0.12);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.2);
  }

  .meta-badge.miss {
    background: rgba(245, 158, 11, 0.1);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.15);
  }

  .meta-latency {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #34d399;
  }

  .meta-sim {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #a78bfa;
  }

  .response-text {
    margin: 0;
    white-space: pre-wrap;
  }

  /* ── Typing Indicator ───────────────────────────────────── */
  .typing {
    display: flex;
    gap: 4px;
    padding: 14px 18px;
    font-size: 22px;
    color: #6366f1;
  }

  .dot-one, .dot-two, .dot-three {
    animation: dotBounce 1.4s ease infinite;
  }
  .dot-two { animation-delay: 0.2s; }
  .dot-three { animation-delay: 0.4s; }

  @keyframes dotBounce {
    0%, 80%, 100% { opacity: 0.3; }
    40% { opacity: 1; }
  }

  /* ── Input Bar ──────────────────────────────────────────── */
  .input-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 24px 20px;
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
    flex-shrink: 0;
  }

  .chat-input {
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

  .chat-input:focus {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08);
  }

  .chat-input::placeholder { color: #3b3d56; }
  .chat-input:disabled { opacity: 0.5; }

  .send-btn {
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    border: none;
    border-radius: 12px;
    color: white;
    cursor: pointer;
    transition: all 0.15s;
    flex-shrink: 0;
  }

  .send-btn:hover:not(:disabled) {
    transform: scale(1.04);
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
  }

  .send-btn:disabled { opacity: 0.35; cursor: not-allowed; }
</style>
