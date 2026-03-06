import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from config import (
    IMPORTANCE_THRESHOLD,
    MEMORY_RETRIEVAL_WEIGHTS,
    RECENCY_DECAY_FACTOR,
    SMALLVILLE_LOCATIONS,
    get_config,
)


def test_get_config_returns_dict():
    """Verify get_config returns a dictionary."""
    config = get_config()
    assert isinstance(config, dict)


def test_get_config_has_required_keys():
    """Verify get_config returns all 12 required keys."""
    config = get_config()
    required_keys = {
        "model", "base_url", "memory_weights", "recency_decay",
        "importance_threshold", "simulation_speed", "tick_duration",
        "sim_days", "num_agents", "locations", "start_date", "start_time"
    }
    assert set(config.keys()) >= required_keys


def test_memory_weights_structure():
    """Verify MEMORY_RETRIEVAL_WEIGHTS has 3 float keys."""
    assert isinstance(MEMORY_RETRIEVAL_WEIGHTS, dict)
    required_keys = {"recency", "importance", "relevance"}
    assert set(MEMORY_RETRIEVAL_WEIGHTS.keys()) == required_keys
    for key, value in MEMORY_RETRIEVAL_WEIGHTS.items():
        assert isinstance(value, (int, float)), f"{key} should be numeric"


def test_locations_count():
    """Verify SMALLVILLE_LOCATIONS has exactly 12 entries."""
    assert isinstance(SMALLVILLE_LOCATIONS, dict)
    assert len(SMALLVILLE_LOCATIONS) == 12


def test_constants_reasonable():
    """Verify constants are in reasonable ranges."""
    # RECENCY_DECAY_FACTOR should be between 0 and 1
    assert 0 < RECENCY_DECAY_FACTOR < 1

    # IMPORTANCE_THRESHOLD should be positive
    assert isinstance(IMPORTANCE_THRESHOLD, (int, float))
    assert IMPORTANCE_THRESHOLD > 0
