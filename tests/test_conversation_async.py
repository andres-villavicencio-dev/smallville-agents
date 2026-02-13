"""Async tests for ConversationManager with mocked LLM."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import AsyncMock, patch
from memory import MemoryStream
from conversation import ConversationManager, Conversation


@pytest.fixture
def two_agent_streams(tmp_db_path):
    """Two memory streams for testing conversations."""
    return {
        "Alice": MemoryStream("Alice", tmp_db_path),
        "Bob": MemoryStream("Bob", tmp_db_path),
    }


@pytest.fixture
def conv_manager():
    return ConversationManager()


@pytest.fixture
def mock_conv_llm():
    """AsyncMock LLM client for conversation tests."""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value="Hello there!")
    mock.score_importance = AsyncMock(return_value=5)
    return mock


@pytest.fixture
def patch_conv_llm(mock_conv_llm):
    """Patch get_llm_client in conversation module."""
    async def fake_get_client(*args, **kwargs):
        return mock_conv_llm

    with patch("conversation.get_llm_client", side_effect=fake_get_client):
        yield mock_conv_llm


@pytest.mark.asyncio
async def test_should_initiate_yes(conv_manager, two_agent_streams, patch_conv_llm):
    """Test should_initiate_conversation returns True when LLM says YES."""
    patch_conv_llm.generate = AsyncMock(return_value="YES")

    result = await conv_manager.should_initiate_conversation(
        initiator="Alice",
        target="Bob",
        context="at the park",
        memory_stream=two_agent_streams["Alice"]
    )
    assert result is True


@pytest.mark.asyncio
async def test_should_initiate_no(conv_manager, two_agent_streams, patch_conv_llm):
    """Test should_initiate_conversation returns False when LLM says NO."""
    patch_conv_llm.generate = AsyncMock(return_value="NO")

    result = await conv_manager.should_initiate_conversation(
        initiator="Alice",
        target="Bob",
        context="at the park",
        memory_stream=two_agent_streams["Alice"]
    )
    assert result is False


@pytest.mark.asyncio
async def test_start_creates_conversation(conv_manager, two_agent_streams, patch_conv_llm):
    """Test start_conversation returns a Conversation with opening message."""
    conversation = await conv_manager.start_conversation(
        agent1="Alice",
        agent2="Bob",
        location="park",
        memory_streams=two_agent_streams
    )

    assert conversation is not None
    assert conversation.agent1 == "Alice"
    assert conversation.agent2 == "Bob"
    assert conversation.location == "park"
    assert len(conversation.turns) >= 1
    assert conversation.active is True


@pytest.mark.asyncio
async def test_continue_adds_turn(conv_manager, two_agent_streams, patch_conv_llm):
    """Test continue_conversation adds a new turn."""
    conversation = await conv_manager.start_conversation(
        agent1="Alice",
        agent2="Bob",
        location="park",
        memory_streams=two_agent_streams
    )

    initial_turns = len(conversation.turns)

    continued = await conv_manager.continue_conversation(
        conversation=conversation,
        memory_streams=two_agent_streams
    )

    assert continued is True
    assert len(conversation.turns) > initial_turns


@pytest.mark.asyncio
async def test_end_deactivates(conv_manager, two_agent_streams, patch_conv_llm):
    """Test end_conversation deactivates and moves to history."""
    conversation = await conv_manager.start_conversation(
        agent1="Alice",
        agent2="Bob",
        location="park",
        memory_streams=two_agent_streams
    )

    await conv_manager.end_conversation(
        conversation=conversation,
        memory_streams=two_agent_streams
    )

    assert conversation.active is False
    key = conv_manager.get_conversation_key("Alice", "Bob")
    assert key not in conv_manager.active_conversations
    assert conversation in conv_manager.conversation_history


@pytest.mark.asyncio
async def test_end_stores_memories(conv_manager, two_agent_streams, patch_conv_llm):
    """Test end_conversation stores memories for both agents."""
    conversation = await conv_manager.start_conversation(
        agent1="Alice",
        agent2="Bob",
        location="park",
        memory_streams=two_agent_streams
    )

    await conv_manager.end_conversation(
        conversation=conversation,
        memory_streams=two_agent_streams
    )

    # Check Alice's memories for "Finished conversation" memory
    alice_memories = two_agent_streams["Alice"].get_memories()
    alice_texts = [m.description for m in alice_memories]
    assert any("finished" in m.lower() or "ended" in m.lower() for m in alice_texts)

    # Check Bob's memories
    bob_memories = two_agent_streams["Bob"].get_memories()
    bob_texts = [m.description for m in bob_memories]
    assert any("finished" in m.lower() or "ended" in m.lower() for m in bob_texts)


@pytest.mark.asyncio
async def test_update_conversations_batch(conv_manager, two_agent_streams, patch_conv_llm):
    """Test update_conversations processes active conversations."""
    conversation = await conv_manager.start_conversation(
        agent1="Alice",
        agent2="Bob",
        location="park",
        memory_streams=two_agent_streams
    )

    initial_turns = len(conversation.turns)

    await conv_manager.update_conversations(memory_streams=two_agent_streams)

    # Should have either continued (more turns) or ended (inactive)
    assert len(conversation.turns) > initial_turns or conversation.active is False


@pytest.mark.asyncio
async def test_start_conversation_adds_memory(conv_manager, two_agent_streams, patch_conv_llm):
    """Test start_conversation adds memories to both agents."""
    conversation = await conv_manager.start_conversation(
        agent1="Alice",
        agent2="Bob",
        location="park",
        memory_streams=two_agent_streams
    )

    # Check Alice's memories
    alice_memories = two_agent_streams["Alice"].get_memories()
    alice_texts = [m.description for m in alice_memories]
    assert any("started" in m.lower() or "talking" in m.lower() or "conversation" in m.lower()
               for m in alice_texts)

    # Check Bob's memories
    bob_memories = two_agent_streams["Bob"].get_memories()
    bob_texts = [m.description for m in bob_memories]
    assert any("started" in m.lower() or "talking" in m.lower() or "conversation" in m.lower()
               for m in bob_texts)
