import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from environment import SmallvilleEnvironment


# A. Initialization Tests

def test_has_all_12_locations(environment):
    assert len(environment.locations) == 12


def test_locations_have_correct_sub_areas(environment):
    lin_home = environment.locations["Lin Family Home"]
    assert set(lin_home.sub_areas) == {"kitchen", "living_room", "bedroom", "bathroom"}


def test_locations_have_objects(environment):
    hobbs_cafe = environment.locations["Hobbs Cafe"]
    assert "coffee machine" in hobbs_cafe.objects

    library = environment.locations["Library"]
    assert "books" in library.objects


def test_initial_agents_empty(environment):
    for location in environment.locations.values():
        assert len(location.current_agents) == 0


# B. Agent Movement Tests

def test_move_agent_success(environment):
    result = environment.move_agent("Alice", "Lin Family Home")
    assert result is True


def test_move_agent_invalid_location(environment):
    result = environment.move_agent("Alice", "Nonexistent Place")
    assert result is False


def test_move_updates_tracking(environment):
    environment.move_agent("Alice", "Lin Family Home")
    location, sub_area = environment.get_agent_location("Alice")
    assert location == "Lin Family Home"
    assert sub_area == "kitchen"  # First sub_area


def test_move_removes_from_old(environment):
    environment.move_agent("Alice", "Lin Family Home")
    environment.move_agent("Alice", "Library")
    lin_home = environment.locations["Lin Family Home"]
    assert "Alice" not in lin_home.current_agents


def test_move_adds_to_new(environment):
    environment.move_agent("Alice", "Library")
    library = environment.locations["Library"]
    assert "Alice" in library.current_agents


def test_move_default_sub_area(environment):
    environment.move_agent("Alice", "Lin Family Home")
    _, sub_area = environment.get_agent_location("Alice")
    assert sub_area == "kitchen"  # First sub_area of Lin Family Home


def test_move_capacity_check(environment):
    environment.locations["Library"].capacity = 1
    result1 = environment.move_agent("Alice", "Library")
    result2 = environment.move_agent("Bob", "Library")
    assert result1 is True
    assert result2 is False


# C. Location Queries Tests

def test_get_agent_location_unplaced(environment):
    location, sub_area = environment.get_agent_location("UnplacedAgent")
    assert location is None
    assert sub_area is None


def test_get_agents_at_location_empty(environment):
    agents = environment.get_agents_at_location("Library")
    assert agents == []


def test_get_agents_at_location_multiple(environment):
    environment.move_agent("Alice", "Library")
    environment.move_agent("Bob", "Library")
    environment.move_agent("Charlie", "Library")
    agents = environment.get_agents_at_location("Library")
    assert len(agents) == 3
    assert set(agents) == {"Alice", "Bob", "Charlie"}


def test_get_agents_at_location_sub_area_filter(environment):
    environment.move_agent("Alice", "Lin Family Home", "kitchen")
    environment.move_agent("Bob", "Lin Family Home", "bedroom")
    kitchen_agents = environment.get_agents_at_location("Lin Family Home", "kitchen")
    assert kitchen_agents == ["Alice"]


def test_get_nearby_agents_excludes_self(environment):
    environment.move_agent("Alice", "Library", "reading_area")
    environment.move_agent("Bob", "Library", "reading_area")
    environment.move_agent("Charlie", "Library", "reading_area")
    nearby = environment.get_nearby_agents("Alice")
    assert len(nearby) == 2
    assert set(nearby) == {"Bob", "Charlie"}


# D. Travel Time Tests

def test_same_location_zero(environment):
    travel_time = environment.calculate_travel_time("Lin Family Home", "Lin Family Home")
    assert travel_time == 0


def test_explicit_pair(environment):
    travel_time = environment.calculate_travel_time("Lin Family Home", "Pharmacy")
    assert travel_time == 2


def test_explicit_pair_reverse(environment):
    travel_time = environment.calculate_travel_time("Pharmacy", "Lin Family Home")
    assert travel_time == 2


def test_same_type_default(environment):
    travel_time = environment.calculate_travel_time("Lin Family Home", "Moreno Family Home")
    assert travel_time == 5


def test_cross_type_default(environment):
    travel_time = environment.calculate_travel_time("The Willows", "Hobbs Cafe")
    assert travel_time == 7


# E. Observation Tests

def test_observe_no_location(environment):
    observations = environment.observe_environment("UnplacedAgent")
    assert observations == []


def test_observe_nearby_agent(environment):
    environment.move_agent("Alice", "Library", "reading_area")
    environment.move_agent("Bob", "Library", "reading_area")
    observations = environment.observe_environment("Alice")
    assert len(observations) > 0
    assert any("Bob" in obs for obs in observations)


def test_observe_format(environment):
    environment.move_agent("Alice", "Library", "reading_area")
    environment.move_agent("Bob", "Library", "reading_area")
    observations = environment.observe_environment("Alice")
    # At least one observation about nearby agent
    agent_observations = [obs for obs in observations if "Bob" in obs]
    assert len(agent_observations) >= 1
    for obs in agent_observations:
        assert "Bob is in Library" in obs


# F. State Serialization Tests

def test_state_structure(environment):
    state = environment.get_environment_state()
    assert "locations" in state
    assert "agent_locations" in state
    assert "agent_sub_areas" in state


def test_save_load_roundtrip(environment):
    environment.move_agent("Alice", "Library", "reading_area")
    environment.move_agent("Bob", "Hobbs Cafe", "seating_area")

    state = environment.get_environment_state()

    new_env = SmallvilleEnvironment()
    new_env.load_environment_state(state)

    alice_loc, alice_sub = new_env.get_agent_location("Alice")
    bob_loc, bob_sub = new_env.get_agent_location("Bob")

    assert alice_loc == "Library"
    assert alice_sub == "reading_area"
    assert bob_loc == "Hobbs Cafe"
    assert bob_sub == "seating_area"


def test_load_clears_old_state(environment):
    environment.move_agent("Alice", "Library")

    state = {"locations": {}, "agent_locations": {}, "agent_sub_areas": {}}
    environment.load_environment_state(state)

    location, sub_area = environment.get_agent_location("Alice")
    assert location is None
    assert sub_area is None


def test_state_agents_as_list(environment):
    environment.move_agent("Alice", "Library")
    environment.move_agent("Bob", "Library")

    state = environment.get_environment_state()

    for location_data in state["locations"].values():
        assert isinstance(location_data["current_agents"], list)
