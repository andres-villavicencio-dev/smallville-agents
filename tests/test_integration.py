import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

import pytest

from agent import GenerativeAgent, PlanItem
from conversation import ConversationManager
from environment import SmallvilleEnvironment
from memory import Memory, MemoryStream
from personas import get_agent_persona, select_agent_subset


def test_agent_creation_loads_persona(tmp_db_path):
    """Verify agent loads persona data and initializes memory stream."""
    agent = GenerativeAgent("John Lin", db_path=tmp_db_path)

    assert agent.persona is not None
    assert "name" in agent.persona
    assert "age" in agent.persona
    assert "occupation" in agent.persona
    assert agent.persona["name"] == "John Lin"
    assert isinstance(agent.memory_stream, MemoryStream)


def test_agent_placed_in_environment(tmp_db_path):
    """Verify agent can be placed in environment and location agrees."""
    agent = GenerativeAgent("John Lin", db_path=tmp_db_path)
    env = SmallvilleEnvironment()

    location = "Lin Family Home"
    env.move_agent("John Lin", location)
    agent.move_to_location(location)

    loc, sub = env.get_agent_location("John Lin")
    assert loc == location
    assert agent.current_location == location


def test_plan_then_execute(tmp_db_path):
    """Verify agent can follow daily plan and mark activities complete."""
    agent = GenerativeAgent("John Lin", db_path=tmp_db_path)

    agent.daily_plan = [
        PlanItem("Open pharmacy", "Pharmacy", datetime(2023, 2, 13, 8, 0), 60),
        PlanItem("Help customers", "Pharmacy", datetime(2023, 2, 13, 9, 15), 120),
    ]

    # At 8:30, first activity should be current
    current = agent.update_current_activity(datetime(2023, 2, 13, 8, 30))
    assert current == "Open pharmacy"
    assert not agent.daily_plan[0].completed

    # At 9:30, second activity should be current and first should be completed
    current = agent.update_current_activity(datetime(2023, 2, 13, 9, 30))
    assert current == "Help customers"
    assert agent.daily_plan[0].completed


@pytest.mark.asyncio
async def test_conversation_flow(tmp_db_path, patch_llm):
    """Verify conversation manager creates conversation and stores memories."""
    agent1 = GenerativeAgent("John Lin", db_path=tmp_db_path)
    agent2 = GenerativeAgent("Mei Lin", db_path=tmp_db_path)

    mgr = ConversationManager()
    streams = {
        "John Lin": agent1.memory_stream,
        "Mei Lin": agent2.memory_stream
    }

    conv = await mgr.start_conversation(
        "John Lin",
        "Mei Lin",
        "Lin Family Home",
        streams
    )

    assert conv is not None
    assert len(conv.turns) >= 1

    # Verify conversation memories were stored for both agents
    john_memories = agent1.memory_stream.get_memories()
    mei_memories = agent2.memory_stream.get_memories()

    john_conv_memories = [m for m in john_memories if "conversation" in m.description.lower()]
    mei_conv_memories = [m for m in mei_memories if "conversation" in m.description.lower() or "talking" in m.description.lower()]

    assert len(john_conv_memories) > 0
    assert len(mei_conv_memories) > 0


def test_memory_retrieval_ranking(tmp_db_path):
    """Verify memory retrieval ranks by relevance, recency, and importance."""
    agent = GenerativeAgent("John Lin", db_path=tmp_db_path)

    base = datetime(2023, 2, 13, 10, 0)
    memories = [
        Memory(
            agent_name="John Lin",
            description="pharmacy prescriptions customers",
            importance_score=8,
            creation_timestamp=base,
            last_access_timestamp=base,
            location="Pharmacy"
        ),
        Memory(
            agent_name="John Lin",
            description="basketball game park sports fun",
            importance_score=3,
            creation_timestamp=base - timedelta(hours=24),
            last_access_timestamp=base - timedelta(hours=24),
            location="Park"
        ),
        Memory(
            agent_name="John Lin",
            description="Valentine party planning event",
            importance_score=9,
            creation_timestamp=base - timedelta(hours=1),
            last_access_timestamp=base - timedelta(hours=1),
            location="Hobbs Cafe"
        ),
    ]

    for mem in memories:
        agent.memory_stream.add_memory(mem)

    # Query for pharmacy-related content
    retrieved = agent.memory_stream.retrieve_memories(
        "pharmacy",
        top_k=3,
        current_time=base
    )

    # retrieve_memories returns List[Tuple[Memory, float]]
    assert len(retrieved) == 3
    assert "pharmacy" in retrieved[0][0].description.lower()


def test_state_save_load(tmp_db_path):
    """Verify agent state can be saved and loaded."""
    agent = GenerativeAgent("John Lin", db_path=tmp_db_path)

    # Set up some state
    agent.move_to_location("Pharmacy")
    agent.daily_plan = [
        PlanItem("Test activity", "Pharmacy", datetime(2023, 2, 13, 8, 0), 60)
    ]
    agent.reflection_count = 5

    # Save state
    state = agent.get_state()

    # Create new agent and load state
    agent2 = GenerativeAgent("John Lin", db_path=tmp_db_path)
    agent2.load_state(state)

    # Verify state matches
    assert agent2.current_location == "Pharmacy"
    assert len(agent2.daily_plan) == 1
    assert agent2.reflection_count == 5


def test_environment_state_with_agents():
    """Verify environment state can be saved and loaded with agents."""
    env = SmallvilleEnvironment()

    # Place 3 agents
    env.move_agent("John Lin", "Lin Family Home")
    env.move_agent("Mei Lin", "Lin Family Home")
    env.move_agent("Eddy Lin", "Pharmacy")

    # Save state
    state = env.get_environment_state()

    # Create new environment and load state
    env2 = SmallvilleEnvironment()
    env2.load_environment_state(state)

    # Verify all agents at correct locations (get_agent_location returns tuple)
    loc1, _ = env2.get_agent_location("John Lin")
    loc2, _ = env2.get_agent_location("Mei Lin")
    loc3, _ = env2.get_agent_location("Eddy Lin")
    assert loc1 == "Lin Family Home"
    assert loc2 == "Lin Family Home"
    assert loc3 == "Pharmacy"


def test_num_agents_subset(tmp_db_path):
    """Verify agent subset selection and environment placement."""
    agent_names = select_agent_subset(5)
    assert len(agent_names) == 5

    env = SmallvilleEnvironment()
    agents = []

    # Create agents and place them at home locations
    for name in agent_names:
        agent = GenerativeAgent(name, db_path=tmp_db_path)
        agents.append(agent)

        home_location = agent.persona.get("home_location", "Park")
        agent.move_to_location(home_location)
        env.move_agent(name, home_location)

    # Verify environment has exactly 5 agents placed
    state = env.get_environment_state()
    assert len(state["agent_locations"]) == 5

    # Verify each agent is at a valid location
    for name in agent_names:
        loc, _ = env.get_agent_location(name)
        assert loc is not None
        assert loc in env.locations
