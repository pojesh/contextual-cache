"""Tests for config validation."""

import pytest
from pydantic import ValidationError

from contextual_cache.config import Settings


class TestConfigValidation:

    def test_valid_defaults(self):
        """Default settings should be valid."""
        s = Settings()
        assert s.cache_capacity > 0
        assert 0 < s.window_pct < 1
        assert 0 < s.target_error_rate < 1

    def test_invalid_cache_capacity(self):
        with pytest.raises(ValidationError, match="cache_capacity"):
            Settings(cache_capacity=0)

    def test_invalid_window_pct_zero(self):
        with pytest.raises(ValidationError, match="window_pct"):
            Settings(window_pct=0.0)

    def test_invalid_window_pct_one(self):
        with pytest.raises(ValidationError, match="window_pct"):
            Settings(window_pct=1.0)

    def test_invalid_target_error_rate(self):
        with pytest.raises(ValidationError, match="target_error_rate"):
            Settings(target_error_rate=0.0)

    def test_invalid_hnsw_m(self):
        with pytest.raises(ValidationError, match="hnsw_m"):
            Settings(hnsw_m=1)

    def test_invalid_embedding_dim(self):
        with pytest.raises(ValidationError, match="embedding_dim"):
            Settings(embedding_dim=0)

    def test_invalid_cms_width(self):
        with pytest.raises(ValidationError, match="cms_width"):
            Settings(cms_width=0)

    def test_invalid_cms_depth(self):
        with pytest.raises(ValidationError, match="cms_depth"):
            Settings(cms_depth=0)

    def test_valid_custom_settings(self):
        s = Settings(
            cache_capacity=100,
            window_pct=0.05,
            target_error_rate=0.01,
            hnsw_m=16,
            embedding_dim=768,
        )
        assert s.cache_capacity == 100
        assert s.window_pct == 0.05
