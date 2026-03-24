"""
Centralized configuration via Pydantic BaseSettings.

Every tunable parameter is exposed as an environment variable.
Load a `.env` file in the project root to override defaults.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class EmbeddingBackend(str, Enum):
    LOCAL = "local"            # sentence-transformers, runs on CPU/GPU
    REMOTE = "remote"          # HTTP embedding service


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI = "openai"


class Settings(BaseSettings):
    """All tunables in one place with sensible defaults."""

    # ── Server ────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ── Cache ─────────────────────────────────────────────────
    cache_capacity: int = 50_000
    window_pct: float = 0.01          # W-TinyLFU window fraction
    default_threshold: float = 0.80   # conformal default τ
    target_error_rate: float = 0.05   # conformal ε
    session_timeout_s: int = 1800     # 30 min
    context_alpha: float = 0.85       # query-dominant fusion weight
    context_decay: float = 0.5        # context vector decay toward current query
    default_ttl_s: int = 3600         # entry TTL in seconds (0 = no expiry)
    ttl_cleanup_interval_s: float = 60.0  # background TTL cleanup interval

    # ── Embedding ─────────────────────────────────────────────
    embedding_backend: EmbeddingBackend = EmbeddingBackend.LOCAL
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_cache_size: int = 10_000

    # ── Vector Index (FAISS) ──────────────────────────────────
    hnsw_m: int = 32                  # HNSW links per node
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    ann_k: int = 5                    # top-k neighbors

    # ── LLM ───────────────────────────────────────────────────
    llm_backend: LLMBackend = LLMBackend.OLLAMA
    llm_model: str = "llama3.2:3b"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: Optional[str] = None
    llm_timeout_s: float = 120.0
    llm_connect_timeout_s: float = 10.0
    llm_read_timeout_s: float = 120.0
    llm_max_tokens: int = 2048
    llm_max_retries: int = 3
    llm_retry_base_delay_s: float = 1.0

    # ── Persistence ─────────────────────────────────────────────
    persistence_enabled: bool = False
    persistence_path: str = "contextual_cache.db"

    # ── Index Rebuild ────────────────────────────────────────
    index_rebuild_interval: int = 500  # rebuild after N removals

    # ── Redis (optional) ──────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = False

    # ── Data Structures ───────────────────────────────────────
    cms_width: int = 4096             # Count-Min Sketch columns
    cms_depth: int = 4                # Count-Min Sketch rows
    lsh_bits: int = 8
    lsh_tables: int = 4

    # ── Bandit ────────────────────────────────────────────────
    bandit_n_arms: int = 10
    bandit_sync_interval_s: float = 60.0
    drift_delta: float = 0.002       # ADWIN sensitivity

    # ── Distributed / Cluster ────────────────────────────────
    cluster_nodes: str = ""             # comma-separated node addresses
    shard_id: str = "node-1"
    virtual_nodes: int = 150            # consistent hashing vnodes

    # ── WAL ──────────────────────────────────────────────────
    wal_enabled: bool = False
    wal_path: str = "contextual_cache.wal"
    wal_checkpoint_interval: int = 1000

    # ── Gossip ───────────────────────────────────────────────
    gossip_enabled: bool = False
    gossip_interval_s: float = 60.0
    gossip_peers: str = ""              # comma-separated peer addresses

    # ── Rate Limiting ────────────────────────────────────────
    rate_limit_rps: float = 100.0
    rate_limit_burst: int = 200
    bulk_query_concurrency: int = 10

    # ── Circuit Breaker ───────────────────────────────────────
    cb_failure_threshold: int = 5
    cb_reset_timeout_s: float = 30.0

    # ── Observability ─────────────────────────────────────────
    metrics_history_size: int = 1000  # rolling window for dashboard

    # ── Validators ──────────────────────────────────────────────

    @field_validator("cache_capacity")
    @classmethod
    def _check_capacity(cls, v: int) -> int:
        if v < 1:
            raise ValueError("cache_capacity must be >= 1")
        return v

    @field_validator("window_pct")
    @classmethod
    def _check_window_pct(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("window_pct must be between 0 and 1 exclusive")
        return v

    @field_validator("target_error_rate")
    @classmethod
    def _check_error_rate(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("target_error_rate must be between 0 and 1")
        return v

    @field_validator("hnsw_m")
    @classmethod
    def _check_hnsw_m(cls, v: int) -> int:
        if v < 2:
            raise ValueError("hnsw_m must be >= 2")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def _check_embedding_dim(cls, v: int) -> int:
        if v < 1:
            raise ValueError("embedding_dim must be >= 1")
        return v

    @field_validator("cms_width")
    @classmethod
    def _check_cms_width(cls, v: int) -> int:
        if v < 1:
            raise ValueError("cms_width must be >= 1")
        return v

    @field_validator("cms_depth")
    @classmethod
    def _check_cms_depth(cls, v: int) -> int:
        if v < 1:
            raise ValueError("cms_depth must be >= 1")
        return v

    model_config = {"env_prefix": "CC_", "env_file": ".env", "extra": "ignore"}


# Singleton – import once, use everywhere
settings = Settings()
