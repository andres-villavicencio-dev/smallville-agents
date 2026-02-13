import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from personas import (
    AGENT_PERSONAS,
    get_all_agent_names,
    get_agent_persona,
    select_agent_subset,
    get_agents_by_location,
    get_agent_relationships,
    format_agent_description
)


def test_get_all_agent_names_count():
    """Verify get_all_agent_names returns exactly 25 names."""
    names = get_all_agent_names()
    assert len(names) == 25


def test_get_all_agent_names_known():
    """Verify get_all_agent_names contains known agents."""
    names = get_all_agent_names()
    assert "John Lin" in names
    assert "Isabella Rodriguez" in names


def test_get_agent_persona_exists():
    """Verify get_agent_persona returns dict with all required keys for existing agent."""
    persona = get_agent_persona("John Lin")
    required_keys = {
        "name", "age", "occupation", "personality", "background",
        "relationships", "daily_routine", "goals", "home_location", "work_location"
    }
    assert isinstance(persona, dict)
    assert set(persona.keys()) >= required_keys


def test_get_agent_persona_unknown():
    """Verify get_agent_persona returns empty dict for unknown agent."""
    persona = get_agent_persona("Unknown Person")
    assert persona == {}


def test_all_personas_have_required_keys():
    """Verify all 25 personas have required fields."""
    required_keys = {
        "name", "age", "occupation", "personality", "background",
        "relationships", "daily_routine", "goals", "home_location", "work_location"
    }
    all_names = get_all_agent_names()
    assert len(all_names) == 25

    for name in all_names:
        persona = get_agent_persona(name)
        assert set(persona.keys()) >= required_keys, f"{name} missing required keys"


def test_select_agent_subset_10():
    """Verify select_agent_subset returns 10 names when requested."""
    names = select_agent_subset(10)
    assert len(names) == 10


def test_select_agent_subset_clamp_low():
    """Verify select_agent_subset clamps to minimum 5."""
    names = select_agent_subset(2)
    assert len(names) == 5


def test_select_agent_subset_clamp_high():
    """Verify select_agent_subset clamps to maximum 25."""
    names = select_agent_subset(100)
    assert len(names) == 25


def test_select_agent_subset_preserves_order():
    """Verify select_agent_subset returns first N names in order."""
    all_names = get_all_agent_names()
    subset = select_agent_subset(5)
    assert subset == all_names[:5]


def test_get_agents_by_location():
    """Verify get_agents_by_location returns agents at Lin Family Home."""
    agents = get_agents_by_location("Lin Family Home")
    assert isinstance(agents, list)
    # John Lin, Mei Lin, and Eddy Lin all have home_location="Lin Family Home"
    assert "John Lin" in agents
    assert "Mei Lin" in agents
    assert "Eddy Lin" in agents


def test_get_agent_relationships():
    """Verify get_agent_relationships returns relationships for John Lin."""
    relationships = get_agent_relationships("John Lin")
    assert isinstance(relationships, dict)
    # John Lin should have relationships with Mei Lin and Eddy Lin
    assert "Mei Lin" in relationships or "Mei Lin" in str(relationships.values())
    assert "Eddy Lin" in relationships or "Eddy Lin" in str(relationships.values())


def test_format_agent_description():
    """Verify format_agent_description contains key info for John Lin."""
    description = format_agent_description("John Lin")
    assert isinstance(description, str)
    persona = get_agent_persona("John Lin")
    # Should contain age, occupation, and personality
    assert str(persona["age"]) in description or "age" in description.lower()
    assert persona["occupation"].lower() in description.lower() or "occupation" in description.lower()


def test_format_agent_description_unknown():
    """Verify format_agent_description returns fallback for unknown agent."""
    description = format_agent_description("Unknown Person")
    assert isinstance(description, str)
    assert "resident of smallville" in description.lower()
    assert "Unknown Person" in description
