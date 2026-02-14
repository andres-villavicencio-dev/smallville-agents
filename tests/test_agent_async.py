import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch


@pytest.mark.asyncio
async def test_observe_stores_memory(test_agent, patch_llm):
    """Observation should create and store a memory with the correct description."""
    await test_agent.observe("saw a cat", "park")

    memories = test_agent.memory_stream.get_memories()
    assert len(memories) == 1
    assert memories[0].description == "saw a cat"
    assert memories[0].memory_type == "observation"
    assert memories[0].location == "park"


@pytest.mark.asyncio
async def test_observe_calls_score_importance(test_agent, patch_llm):
    """Observe should call LLM to score the observation's importance."""
    await test_agent.observe("saw a cat", "park")

    patch_llm.score_importance.assert_awaited()
    # Verify it was called with the observation text
    call_args = patch_llm.score_importance.call_args
    assert "saw a cat" in str(call_args)


@pytest.mark.asyncio
async def test_observe_error_fallback(test_agent, patch_llm):
    """If importance scoring fails, should fallback to importance=5."""
    # Make score_importance raise an exception
    patch_llm.score_importance.side_effect = Exception("LLM error")

    await test_agent.observe("error test", "location")

    memories = test_agent.memory_stream.get_memories()
    assert len(memories) == 1
    assert memories[0].importance_score == 5  # fallback value


@pytest.mark.asyncio
async def test_observe_triggers_reflection(test_agent, patch_llm):
    """High importance accumulation should trigger reflection."""
    # Set last reflection time far in the past to pass the cooldown check
    test_agent.last_reflection_time = datetime.now() - timedelta(seconds=400)

    # Add at least 3 memories first (so reflect doesn't bail on <3 memories)
    await test_agent.observe("memory 1", "location")
    await test_agent.observe("memory 2", "location")
    await test_agent.observe("memory 3", "location")

    initial_reflection_count = test_agent.reflection_count

    # Mock get_importance_since to return value >150 to trigger reflection
    with patch.object(test_agent.memory_stream, 'get_importance_since', return_value=200):
        await test_agent.observe("high importance observation", "location")

    # Reflection should have been triggered
    assert test_agent.reflection_count > initial_reflection_count


@pytest.mark.asyncio
async def test_reflect_returns_reflection_memories(test_agent, patch_llm):
    """Reflect should generate and return reflection memories."""
    # Add some memories first (need at least 3)
    for i in range(5):
        await test_agent.observe(f"observation {i}", "location")

    reflections = await test_agent.reflect()

    # Mock generates 3 questions, should get 3 reflections
    assert len(reflections) > 0
    for reflection in reflections:
        assert reflection.memory_type == "reflection"
        assert reflection.importance_score == 8


@pytest.mark.asyncio
async def test_reflect_too_few_memories(test_agent, patch_llm):
    """Reflect with <3 memories should return empty list."""
    # Fresh agent with no memories
    reflections = await test_agent.reflect()
    assert reflections == []

    # Add 1 memory, still not enough
    await test_agent.observe("single observation", "location")
    reflections = await test_agent.reflect()
    assert reflections == []


@pytest.mark.asyncio
async def test_reflect_increments_count(test_agent, patch_llm):
    """Reflection should increment reflection_count."""
    # Add enough memories
    for i in range(5):
        await test_agent.observe(f"observation {i}", "location")

    initial_count = test_agent.reflection_count
    await test_agent.reflect()

    assert test_agent.reflection_count == initial_count + 1


@pytest.mark.asyncio
async def test_reflect_updates_timestamp(test_agent, patch_llm):
    """Reflection should update last_reflection_time."""
    # Add enough memories
    for i in range(5):
        await test_agent.observe(f"observation {i}", "location")

    old_time = test_agent.last_reflection_time
    await test_agent.reflect()

    assert test_agent.last_reflection_time > old_time


@pytest.mark.asyncio
async def test_plan_creates_items(test_agent, patch_llm):
    """Planning should create and store plan items."""
    await test_agent.plan_daily_schedule(datetime(2023, 2, 13))

    assert len(test_agent.daily_plan) > 0
    # Verify we got plan items
    for item in test_agent.daily_plan:
        assert hasattr(item, 'description')
        assert hasattr(item, 'start_time')


@pytest.mark.asyncio
async def test_plan_stores_plan_memory(test_agent, patch_llm):
    """Planning should store a plan-type memory."""
    await test_agent.plan_daily_schedule(datetime(2023, 2, 13))

    memories = test_agent.memory_stream.get_memories()
    plan_memories = [m for m in memories if m.memory_type == "plan"]

    assert len(plan_memories) > 0
    assert plan_memories[0].importance_score == 7


@pytest.mark.asyncio
async def test_plan_decomposes_long_items(test_agent, patch_llm):
    """Long plan items (>30 min) should be decomposed."""
    await test_agent.plan_daily_schedule(datetime(2023, 2, 13))

    # The mock plan has "work at pharmacy" which should be >30min and trigger decomposition
    patch_llm.decompose_plan_item.assert_awaited()


@pytest.mark.asyncio
async def test_plan_error_returns_empty(test_agent, patch_llm):
    """If planning fails, should return empty list."""
    # Make generate_daily_plan raise an exception
    patch_llm.generate_daily_plan.side_effect = Exception("Planning error")

    result = await test_agent.plan_daily_schedule(datetime(2023, 2, 13))

    assert result == []
    assert len(test_agent.daily_plan) == 0
