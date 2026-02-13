import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import sqlite3
from datetime import datetime, timedelta
from memory import Memory, MemoryStream


# ============================================================================
# A. Memory Dataclass (6 tests)
# ============================================================================

def test_memory_defaults():
    """Test that Memory dataclass has correct default values"""
    mem = Memory(
        agent_name="Agent",
        description="test"
    )
    assert mem.id is None
    assert mem.memory_type == "observation"
    assert mem.importance_score == 5
    assert mem.source_memory_ids == []


def test_memory_post_init_timestamps():
    """Test that last_access_timestamp defaults to creation_timestamp"""
    mem = Memory(
        agent_name="Agent",
        description="test",
        creation_timestamp=datetime(2023, 1, 1, 12, 0, 0)
    )
    assert mem.last_access_timestamp == mem.creation_timestamp
    assert mem.last_access_timestamp == datetime(2023, 1, 1, 12, 0, 0)


def test_memory_post_init_source_ids():
    """Test that source_memory_ids defaults to [] and is not shared mutable"""
    mem1 = Memory(agent_name="Agent1", description="test1")
    mem2 = Memory(agent_name="Agent2", description="test2")

    # Modify one's list
    mem1.source_memory_ids.append(42)

    # Other should be unaffected
    assert mem1.source_memory_ids == [42]
    assert mem2.source_memory_ids == []


def test_memory_to_dict():
    """Test that to_dict() returns all fields with ISO timestamp strings"""
    creation = datetime(2023, 2, 13, 8, 0, 0)
    access = datetime(2023, 2, 13, 10, 0, 0)

    mem = Memory(
        id=5,
        agent_name="TestAgent",
        description="A test memory",
        memory_type="reflection",
        importance_score=8,
        creation_timestamp=creation,
        last_access_timestamp=access,
        location="Park",
        source_memory_ids=[1, 2, 3]
    )

    d = mem.to_dict()

    assert d["id"] == 5
    assert d["agent_name"] == "TestAgent"
    assert d["description"] == "A test memory"
    assert d["memory_type"] == "reflection"
    assert d["importance_score"] == 8
    assert d["creation_timestamp"] == creation.isoformat()
    assert d["last_access_timestamp"] == access.isoformat()
    assert d["location"] == "Park"
    assert d["source_memory_ids"] == [1, 2, 3]


def test_memory_from_dict_roundtrip():
    """Test that from_dict(to_dict()) produces equivalent Memory"""
    original = Memory(
        id=10,
        agent_name="Agent",
        description="Original memory",
        memory_type="plan",
        importance_score=7,
        creation_timestamp=datetime(2023, 5, 15, 14, 30, 0),
        last_access_timestamp=datetime(2023, 5, 15, 16, 0, 0),
        location="Home"
    )

    d = original.to_dict()
    restored = Memory.from_dict(d)

    assert restored.id == original.id
    assert restored.agent_name == original.agent_name
    assert restored.description == original.description
    assert restored.memory_type == original.memory_type
    assert restored.importance_score == original.importance_score
    assert restored.creation_timestamp == original.creation_timestamp
    assert restored.last_access_timestamp == original.last_access_timestamp
    assert restored.location == original.location
    assert restored.source_memory_ids == original.source_memory_ids


def test_memory_from_dict_with_source_ids():
    """Test that from_dict roundtrip preserves source_memory_ids"""
    original = Memory(
        agent_name="Agent",
        description="Memory with sources",
        source_memory_ids=[1, 2, 3]
    )

    d = original.to_dict()
    restored = Memory.from_dict(d)

    assert restored.source_memory_ids == [1, 2, 3]


# ============================================================================
# B. MemoryStream DB Operations (9 tests)
# ============================================================================

def test_init_creates_tables(tmp_db_path):
    """Test that MemoryStream initialization creates necessary tables"""
    stream = MemoryStream("TestAgent", tmp_db_path)

    conn = sqlite3.connect(tmp_db_path)
    cursor = conn.cursor()

    # Check memories table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
    assert cursor.fetchone() is not None

    # Check memories_fts virtual table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'")
    assert cursor.fetchone() is not None

    conn.close()


def test_add_memory_returns_positive_id(memory_stream, sample_memory):
    """Test that add_memory returns a positive integer ID"""
    mem = sample_memory()
    mem_id = memory_stream.add_memory(mem)

    assert isinstance(mem_id, int)
    assert mem_id > 0


def test_add_memory_persists(memory_stream, sample_memory):
    """Test that added memory can be retrieved with correct fields"""
    mem = sample_memory(
        description="Persistent memory",
        memory_type="reflection",
        importance_score=8,
        location="Library"
    )
    mem_id = memory_stream.add_memory(mem)

    retrieved = memory_stream.get_memories()
    assert len(retrieved) == 1
    assert retrieved[0].id == mem_id
    assert retrieved[0].description == "Persistent memory"
    assert retrieved[0].memory_type == "reflection"
    assert retrieved[0].importance_score == 8
    assert retrieved[0].location == "Library"


def test_add_multiple_memories_ordered(memory_stream, sample_memory):
    """Test that get_memories returns memories in DESC order by timestamp"""
    base_time = datetime(2023, 1, 1, 12, 0, 0)

    for i in range(5):
        mem = sample_memory(
            description=f"Memory {i}",
            creation_timestamp=base_time + timedelta(hours=i)
        )
        memory_stream.add_memory(mem)

    memories = memory_stream.get_memories()
    assert len(memories) == 5

    # Should be in DESC order (newest first)
    for i in range(4):
        assert memories[i].creation_timestamp > memories[i+1].creation_timestamp


def test_get_memories_limit(memory_stream, sample_memory):
    """Test that get_memories limit parameter works"""
    for i in range(10):
        mem = sample_memory(description=f"Memory {i}")
        memory_stream.add_memory(mem)

    memories = memory_stream.get_memories(limit=3)
    assert len(memories) == 3


def test_get_memories_by_type(memory_stream, sample_memory):
    """Test that get_memories filters by memory_type"""
    memory_stream.add_memory(sample_memory(memory_type="observation"))
    memory_stream.add_memory(sample_memory(memory_type="observation"))
    memory_stream.add_memory(sample_memory(memory_type="reflection"))
    memory_stream.add_memory(sample_memory(memory_type="plan"))
    memory_stream.add_memory(sample_memory(memory_type="reflection"))

    observations = memory_stream.get_memories(memory_type="observation")
    reflections = memory_stream.get_memories(memory_type="reflection")
    plans = memory_stream.get_memories(memory_type="plan")

    assert len(observations) == 2
    assert len(reflections) == 2
    assert len(plans) == 1

    assert all(m.memory_type == "observation" for m in observations)
    assert all(m.memory_type == "reflection" for m in reflections)
    assert all(m.memory_type == "plan" for m in plans)


def test_get_memories_agent_isolation(tmp_db_path, sample_memory):
    """Test that different agents' memories are isolated"""
    stream1 = MemoryStream("Agent1", tmp_db_path)
    stream2 = MemoryStream("Agent2", tmp_db_path)

    # Add memories to each agent
    mem1 = sample_memory(description="Agent1's memory")
    mem1.agent_name = "Agent1"
    stream1.add_memory(mem1)

    mem2 = sample_memory(description="Agent2's memory")
    mem2.agent_name = "Agent2"
    stream2.add_memory(mem2)

    # Each should see only their own memories
    agent1_memories = stream1.get_memories()
    agent2_memories = stream2.get_memories()

    assert len(agent1_memories) == 1
    assert len(agent2_memories) == 1
    assert agent1_memories[0].description == "Agent1's memory"
    assert agent2_memories[0].description == "Agent2's memory"


def test_add_memory_with_source_ids(memory_stream, sample_memory):
    """Test that source_memory_ids are stored and retrieved correctly"""
    mem = sample_memory(source_memory_ids=[1, 2])
    mem_id = memory_stream.add_memory(mem)

    retrieved = memory_stream.get_memories()
    assert len(retrieved) == 1
    assert retrieved[0].source_memory_ids == [1, 2]


def test_get_memories_empty(memory_stream):
    """Test that get_memories returns empty list for fresh stream"""
    memories = memory_stream.get_memories()
    assert memories == []


# ============================================================================
# C. Recency Score (5 tests)
# ============================================================================

def test_recency_zero_hours(memory_stream, sample_memory):
    """Test that recency score is 1.0 when access time equals current time"""
    current = datetime.now()
    mem = sample_memory(last_access_timestamp=current)

    score = memory_stream._calculate_recency_score(mem, current)
    assert score == pytest.approx(1.0)


def test_recency_one_hour(memory_stream, sample_memory):
    """Test that recency score is ~0.99 after one hour"""
    current = datetime.now()
    one_hour_ago = current - timedelta(hours=1)
    mem = sample_memory(last_access_timestamp=one_hour_ago)

    score = memory_stream._calculate_recency_score(mem, current)
    assert score == pytest.approx(0.99, abs=0.001)


def test_recency_24_hours(memory_stream, sample_memory):
    """Test that recency score is ~0.7854 after 24 hours (0.99^24)"""
    current = datetime.now()
    one_day_ago = current - timedelta(hours=24)
    mem = sample_memory(last_access_timestamp=one_day_ago)

    score = memory_stream._calculate_recency_score(mem, current)
    expected = 0.99 ** 24
    assert score == pytest.approx(expected, abs=0.001)


def test_recency_one_week(memory_stream, sample_memory):
    """Test that recency score is ~0.1845 after one week (0.99^168)"""
    current = datetime.now()
    one_week_ago = current - timedelta(weeks=1)
    mem = sample_memory(last_access_timestamp=one_week_ago)

    score = memory_stream._calculate_recency_score(mem, current)
    expected = 0.99 ** 168
    assert score == pytest.approx(expected, abs=0.001)


def test_recency_monotonically_decreasing(memory_stream, sample_memory):
    """Test that recency scores strictly decrease with age"""
    current = datetime.now()
    ages_hours = [0, 1, 6, 24, 168]
    scores = []

    for hours in ages_hours:
        timestamp = current - timedelta(hours=hours)
        mem = sample_memory(last_access_timestamp=timestamp)
        score = memory_stream._calculate_recency_score(mem, current)
        scores.append(score)

    # Verify strictly decreasing
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1]


# ============================================================================
# D. Importance Score (4 tests)
# ============================================================================

def test_importance_min(memory_stream, sample_memory):
    """Test that importance score of 1 maps to 0.0"""
    mem = sample_memory(importance_score=1)
    score = memory_stream._calculate_importance_score(mem)
    assert score == pytest.approx(0.0)


def test_importance_max(memory_stream, sample_memory):
    """Test that importance score of 10 maps to 1.0"""
    mem = sample_memory(importance_score=10)
    score = memory_stream._calculate_importance_score(mem)
    assert score == pytest.approx(1.0)


def test_importance_mid(memory_stream, sample_memory):
    """Test that importance score of 5 maps to ~0.4444"""
    mem = sample_memory(importance_score=5)
    score = memory_stream._calculate_importance_score(mem)
    expected = (5 - 1) / 9.0
    assert score == pytest.approx(expected, abs=0.001)


def test_importance_all_values_in_range(memory_stream, sample_memory):
    """Test that all importance values map to [0,1] and are monotonically increasing"""
    scores = []
    for importance in range(1, 11):
        mem = sample_memory(importance_score=importance)
        score = memory_stream._calculate_importance_score(mem)
        scores.append(score)

        # Score should be in valid range
        assert 0.0 <= score <= 1.0

    # Verify monotonically increasing
    for i in range(len(scores) - 1):
        assert scores[i] < scores[i + 1]


# ============================================================================
# E. Relevance Score (4 tests)
# ============================================================================

def test_relevance_empty_query(memory_stream, sample_memory):
    """Test that relevance score returns 0.5 for empty query"""
    mem = sample_memory(description="Some text")
    score = memory_stream._calculate_relevance_score(mem, "")
    assert score == 0.5


def test_relevance_no_vectors(memory_stream, sample_memory):
    """Test that relevance score returns 0.5 when no vectors exist"""
    mem = sample_memory(description="Test memory")
    # Don't add memory to stream, so no vectors exist
    score = memory_stream._calculate_relevance_score(mem, "test query")
    assert score == 0.5


def test_relevance_exact_match(memory_stream, sample_memory):
    """Test that exact match query produces high relevance score (>0.7)"""
    description = "pharmacy prescriptions medicine"
    mem = sample_memory(description=description)
    memory_stream.add_memory(mem)

    # Retrieve stored memory (has DB id needed for vector lookup)
    stored = memory_stream.get_memories()[0]
    score = memory_stream._calculate_relevance_score(stored, description)
    assert score > 0.7


def test_relevance_unrelated(memory_stream, sample_memory):
    """Test that unrelated memories have different relevance scores"""
    mem1 = sample_memory(description="pharmacy prescriptions medicine")
    mem2 = sample_memory(description="basketball game sports")

    memory_stream.add_memory(mem1)
    memory_stream.add_memory(mem2)

    # Retrieve stored memories (need DB ids for vector lookup)
    stored = memory_stream.get_memories()
    # Memories come back DESC by timestamp; both have same timestamp so order may vary
    stored_by_desc = {m.description: m for m in stored}
    stored1 = stored_by_desc["pharmacy prescriptions medicine"]
    stored2 = stored_by_desc["basketball game sports"]

    query = "pharmacy"
    score1 = memory_stream._calculate_relevance_score(stored1, query)
    score2 = memory_stream._calculate_relevance_score(stored2, query)

    assert score1 > score2


# ============================================================================
# F. Combined Retrieval (8 tests)
# ============================================================================

def test_retrieve_empty_stream(memory_stream):
    """Test that retrieve_memories returns empty list for fresh stream"""
    results = memory_stream.retrieve_memories("test query")
    assert results == []


def test_retrieve_returns_tuples(populated_memory_stream):
    """Test that retrieve_memories returns list of (Memory, float) tuples"""
    results = populated_memory_stream.retrieve_memories("test")

    assert isinstance(results, list)
    assert len(results) > 0

    for item in results:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], Memory)
        assert isinstance(item[1], float)


def test_retrieve_top_k(populated_memory_stream):
    """Test that retrieve_memories respects top_k parameter"""
    results = populated_memory_stream.retrieve_memories("test", top_k=5)
    assert len(results) == 5


def test_retrieve_scores_in_range(populated_memory_stream):
    """Test that all retrieval scores are in [0, 1]"""
    results = populated_memory_stream.retrieve_memories("test")

    for _, score in results:
        assert 0.0 <= score <= 1.0


def test_retrieve_sorted_descending(populated_memory_stream):
    """Test that retrieve_memories returns results sorted by score descending"""
    results = populated_memory_stream.retrieve_memories("test", top_k=10)

    scores = [score for _, score in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_retrieve_updates_access_time(populated_memory_stream):
    """Test that retrieve_memories updates last_access_timestamp"""
    # Get original access times
    original_memories = populated_memory_stream.get_memories()
    original_times = {m.id: m.last_access_timestamp for m in original_memories}

    # Small delay to ensure timestamp difference
    import time
    time.sleep(0.01)

    # Retrieve memories
    populated_memory_stream.retrieve_memories("test")

    # Check that access times were updated
    updated_memories = populated_memory_stream.get_memories()
    for mem in updated_memories:
        assert mem.last_access_timestamp > original_times[mem.id]


def test_retrieve_recency_dominates(memory_stream, sample_memory):
    """Test that recent memories rank higher than old ones with same importance"""
    current = datetime.now()

    # Add recent memory
    recent = sample_memory(
        description="generic memory text",
        importance_score=5,
        creation_timestamp=current,
        last_access_timestamp=current
    )
    recent_id = memory_stream.add_memory(recent)

    # Add old memory
    old = sample_memory(
        description="generic memory text",
        importance_score=5,
        creation_timestamp=current - timedelta(days=7),
        last_access_timestamp=current - timedelta(days=7)
    )
    old_id = memory_stream.add_memory(old)

    results = memory_stream.retrieve_memories("generic", top_k=2, current_time=current)

    # Recent should rank higher
    assert results[0][0].id == recent_id
    assert results[1][0].id == old_id


def test_retrieve_importance_dominates(memory_stream, sample_memory):
    """Test that high-importance memories rank higher than low-importance ones"""
    current = datetime.now()

    # Add high importance memory
    high = sample_memory(
        description="important memory",
        importance_score=10,
        creation_timestamp=current,
        last_access_timestamp=current
    )
    high_id = memory_stream.add_memory(high)

    # Add low importance memory
    low = sample_memory(
        description="unimportant memory",
        importance_score=1,
        creation_timestamp=current,
        last_access_timestamp=current
    )
    low_id = memory_stream.add_memory(low)

    results = memory_stream.retrieve_memories("memory", top_k=2, current_time=current)

    # High importance should rank higher
    assert results[0][0].id == high_id
    assert results[1][0].id == low_id


# ============================================================================
# G. FTS Search (3 tests)
# ============================================================================

def test_fts_basic_match(memory_stream, sample_memory):
    """Test that FTS search finds correct memory by keyword"""
    memory_stream.add_memory(sample_memory(description="pharmacy prescriptions medicine"))
    memory_stream.add_memory(sample_memory(description="basketball game sports"))
    memory_stream.add_memory(sample_memory(description="coffee cafe morning"))

    results = memory_stream.search_memories_fts("pharmacy")

    assert len(results) >= 1
    assert any("pharmacy" in m.description.lower() for m in results)


def test_fts_limit(memory_stream, sample_memory):
    """Test that FTS search respects limit parameter"""
    for i in range(30):
        memory_stream.add_memory(sample_memory(description=f"test memory number {i}"))

    results = memory_stream.search_memories_fts("test", limit=5)
    assert len(results) == 5


def test_fts_agent_isolation(tmp_db_path, sample_memory):
    """Test that FTS search only returns current agent's memories"""
    stream1 = MemoryStream("Agent1", tmp_db_path)
    stream2 = MemoryStream("Agent2", tmp_db_path)

    mem1 = sample_memory(description="pharmacy search test")
    mem1.agent_name = "Agent1"
    stream1.add_memory(mem1)

    mem2 = sample_memory(description="pharmacy search test")
    mem2.agent_name = "Agent2"
    stream2.add_memory(mem2)

    results1 = stream1.search_memories_fts("pharmacy")
    results2 = stream2.search_memories_fts("pharmacy")

    assert len(results1) == 1
    assert len(results2) == 1
    assert results1[0].agent_name == "Agent1"
    assert results2[0].agent_name == "Agent2"


# ============================================================================
# H. Recent Importance Sum (3 tests)
# ============================================================================

def test_recent_sum_basic(memory_stream, sample_memory):
    """Test basic recent importance sum calculation"""
    current = datetime.now()

    memory_stream.add_memory(sample_memory(
        importance_score=5,
        creation_timestamp=current
    ))
    memory_stream.add_memory(sample_memory(
        importance_score=7,
        creation_timestamp=current
    ))
    memory_stream.add_memory(sample_memory(
        importance_score=3,
        creation_timestamp=current
    ))

    total = memory_stream.get_recent_importance_sum(hours=24)
    assert total == 15


def test_recent_sum_excludes_old(memory_stream, sample_memory):
    """Test that old memories are excluded from recent importance sum"""
    current = datetime.now()

    # Add recent memory
    memory_stream.add_memory(sample_memory(
        importance_score=5,
        creation_timestamp=current
    ))

    # Add old memory (48 hours ago)
    memory_stream.add_memory(sample_memory(
        importance_score=10,
        creation_timestamp=current - timedelta(hours=48)
    ))

    # Only recent memory should count in 24h window
    total = memory_stream.get_recent_importance_sum(hours=24)
    assert total == 5


def test_recent_sum_exclude_types(memory_stream, sample_memory):
    """Test that exclude_types parameter filters out specified memory types"""
    current = datetime.now()

    memory_stream.add_memory(sample_memory(
        memory_type="observation",
        importance_score=5,
        creation_timestamp=current
    ))
    memory_stream.add_memory(sample_memory(
        memory_type="reflection",
        importance_score=8,
        creation_timestamp=current
    ))
    memory_stream.add_memory(sample_memory(
        memory_type="observation",
        importance_score=3,
        creation_timestamp=current
    ))

    # Exclude reflections
    total = memory_stream.get_recent_importance_sum(
        hours=24,
        exclude_types=["reflection"]
    )

    # Should only count observations: 5 + 3 = 8
    assert total == 8
