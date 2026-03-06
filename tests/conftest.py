"""Shared fixtures for the generative-agents test suite."""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation import Conversation, ConversationManager
from environment import SmallvilleEnvironment
from memory import Memory, MemoryStream


@pytest.fixture
def tmp_db_path(tmp_path):
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_memories.db")


@pytest.fixture
def memory_stream(tmp_db_path):
    """Provide a fresh MemoryStream for 'TestAgent'."""
    return MemoryStream("TestAgent", tmp_db_path)


@pytest.fixture
def sample_memory():
    """Factory fixture: returns a function that creates Memory objects with sensible defaults."""
    def make_memory(**overrides):
        defaults = {
            "agent_name": "TestAgent",
            "description": "Saw a cat in the park",
            "memory_type": "observation",
            "importance_score": 5,
            "creation_timestamp": datetime(2023, 2, 13, 10, 0, 0),
            "last_access_timestamp": datetime(2023, 2, 13, 10, 0, 0),
            "location": "Johnson Park",
            "source_memory_ids": [],
        }
        defaults.update(overrides)
        return Memory(**defaults)
    return make_memory


@pytest.fixture
def populated_memory_stream(memory_stream, sample_memory):
    """MemoryStream with 10+ diverse memories of varying importance, timestamps, and types."""
    base_time = datetime(2023, 2, 13, 8, 0, 0)
    memories_data = [
        ("Woke up and had breakfast", "observation", 2, 0, "Lin Family Home"),
        ("Walked to the pharmacy", "observation", 3, 30, "Pharmacy"),
        ("Opened the pharmacy for the day", "observation", 4, 60, "Pharmacy"),
        ("Helped a customer with a prescription", "observation", 5, 90, "Pharmacy"),
        ("Noticed Isabella Rodriguez setting up for an event", "observation", 7, 120, "Hobbs Cafe"),
        ("I think Isabella is planning something special for Valentine's Day", "reflection", 8, 150, "Pharmacy"),
        ("Plan: attend Valentine's Day party at Hobbs Cafe", "plan", 7, 180, "Hobbs Cafe"),
        ("Tom Moreno stopped by the pharmacy", "observation", 6, 210, "Pharmacy"),
        ("Had a conversation with Tom about hardware supplies", "observation", 5, 240, "Pharmacy"),
        ("The weather is nice today", "observation", 1, 270, "Johnson Park"),
        ("Met Sarah Chen at the library", "observation", 6, 300, "Library"),
        ("Community bonds seem strong in Smallville", "reflection", 8, 330, "Library"),
    ]

    for desc, mem_type, importance, minutes_offset, location in memories_data:
        mem = sample_memory(
            description=desc,
            memory_type=mem_type,
            importance_score=importance,
            creation_timestamp=base_time + timedelta(minutes=minutes_offset),
            last_access_timestamp=base_time + timedelta(minutes=minutes_offset),
            location=location,
        )
        memory_stream.add_memory(mem)

    return memory_stream


@pytest.fixture
def environment():
    """Provide a fresh SmallvilleEnvironment."""
    return SmallvilleEnvironment()


@pytest.fixture
def mock_llm_client():
    """Provide an AsyncMock of OllamaClient with pre-configured returns."""
    mock = AsyncMock()
    mock.score_importance = AsyncMock(return_value=5)
    mock.generate_reflection_questions = AsyncMock(
        return_value=["What are the most important events?", "How do relationships evolve?", "What goals are being pursued?"]
    )
    mock.generate_reflection = AsyncMock(return_value="A reflection about recent events.")
    mock.generate_daily_plan = AsyncMock(
        return_value="1) wake up at 8:00 am\n2) work at pharmacy at 9:00 am\n3) lunch at 12:00 pm\n4) afternoon tasks at 1:00 pm\n5) close pharmacy at 6:00 pm"
    )
    mock.decompose_plan_item = AsyncMock(
        return_value=["- Sub-action 1: prepare workspace", "- Sub-action 2: begin main task"]
    )
    mock.should_initiate_conversation = AsyncMock(return_value=True)
    mock.generate_conversation_response = AsyncMock(return_value="Hello!")
    mock.generate = AsyncMock(return_value="YES")
    return mock


@pytest.fixture
def patch_llm(mock_llm_client):
    """Patch get_llm_client in both agent and conversation modules to return the mock."""
    async def fake_get_client(*args, **kwargs):
        return mock_llm_client

    with patch("agent.get_llm_client", side_effect=fake_get_client), \
         patch("conversation.get_llm_client", side_effect=fake_get_client):
        yield mock_llm_client


@pytest.fixture
def test_agent(tmp_db_path, patch_llm):
    """Provide a GenerativeAgent('John Lin') with patched LLM and temp DB."""
    from agent import GenerativeAgent
    return GenerativeAgent("John Lin", db_path=tmp_db_path)


@pytest.fixture
def conversation_manager():
    """Provide a fresh ConversationManager."""
    return ConversationManager()


@pytest.fixture
def fixed_now():
    """Deterministic datetime: 2023-02-13 10:00:00."""
    return datetime(2023, 2, 13, 10, 0, 0)
