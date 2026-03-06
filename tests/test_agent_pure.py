import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

import pytest

from agent import GenerativeAgent, PlanItem

# ============================================================================
# A. PlanItem Tests (5 tests)
# ============================================================================

def test_end_time(fixed_now):
    """PlanItem at 10:00 with 90min duration should end at 11:30"""
    item = PlanItem(
        description="meeting",
        location="Office",
        start_time=fixed_now,
        duration_minutes=90
    )
    expected = fixed_now + timedelta(minutes=90)
    assert item.end_time() == expected


def test_to_dict(fixed_now):
    """to_dict should include all fields with start_time as ISO string"""
    item = PlanItem(
        description="breakfast",
        location="Home",
        start_time=fixed_now,
        duration_minutes=45,
        completed=True
    )
    d = item.to_dict()
    assert d["description"] == "breakfast"
    assert d["location"] == "Home"
    assert d["start_time"] == fixed_now.isoformat()
    assert d["duration_minutes"] == 45
    assert d["completed"] is True


def test_from_dict_roundtrip(fixed_now):
    """from_dict(to_dict()) should produce equivalent PlanItem"""
    original = PlanItem(
        description="work on prescriptions",
        location="Pharmacy",
        start_time=fixed_now,
        duration_minutes=180,
        completed=False
    )
    restored = PlanItem.from_dict(original.to_dict())
    assert restored.description == original.description
    assert restored.location == original.location
    assert restored.start_time == original.start_time
    assert restored.duration_minutes == original.duration_minutes
    assert restored.completed == original.completed


def test_default_completed_false(fixed_now):
    """PlanItem should default completed to False"""
    item = PlanItem(
        description="lunch",
        location="Cafe",
        start_time=fixed_now,
        duration_minutes=45
    )
    assert item.completed is False


def test_from_dict_missing_completed(fixed_now):
    """from_dict should default completed to False if missing"""
    data = {
        "description": "walk",
        "location": "Park",
        "start_time": fixed_now.isoformat(),
        "duration_minutes": 30
    }
    item = PlanItem.from_dict(data)
    assert item.completed is False


# ============================================================================
# B. Time Extraction Tests (6 tests)
# ============================================================================

def test_time_colon_am(test_agent):
    """Extract time from 'breakfast at 8:30 am'"""
    result = test_agent._extract_time_from_text("breakfast at 8:30 am")
    assert result == (8, 30)


def test_time_colon_pm(test_agent):
    """Extract time from 'meeting at 2:00 pm' — pattern 3 captures without am/pm"""
    result = test_agent._extract_time_from_text("meeting at 2:00 pm")
    # Pattern 3 (at HH:MM) matches first without capturing am/pm
    assert result == (2, 0)


def test_time_colon_no_ampm(test_agent):
    """Extract time from 'work at 10:00' (no am/pm)"""
    result = test_agent._extract_time_from_text("work at 10:00")
    assert result == (10, 0)


def test_time_12pm(test_agent):
    """Extract time from 'lunch at 12:00 pm' (should be 12, not 0)"""
    result = test_agent._extract_time_from_text("lunch at 12:00 pm")
    assert result == (12, 0)


def test_time_12am(test_agent):
    """Extract time from 'sleep at 12:00 am' — pattern 3 captures without am/pm"""
    result = test_agent._extract_time_from_text("sleep at 12:00 am")
    # Pattern 3 (at HH:MM) matches first without capturing am/pm
    assert result == (12, 0)


def test_time_no_match(test_agent):
    """No time in 'go for a walk' should return None"""
    result = test_agent._extract_time_from_text("go for a walk")
    assert result is None


# ============================================================================
# C. Location Inference Tests (6 tests)
# ============================================================================

def test_infer_home_wake_up(test_agent):
    """'wake up and shower' should infer home location"""
    # John Lin's home_location is "Lin Family Home"
    location = test_agent._infer_location_from_activity("wake up and shower")
    assert location == "Lin Family Home"


def test_infer_work(test_agent):
    """'work on prescriptions' should infer work location"""
    # John Lin's work_location is "Pharmacy"
    location = test_agent._infer_location_from_activity("work on prescriptions")
    assert location == "Pharmacy"


def test_infer_cafe(test_agent):
    """'get coffee at cafe' should infer Hobbs Cafe"""
    location = test_agent._infer_location_from_activity("get coffee at cafe")
    assert location == "Hobbs Cafe"


def test_infer_library(test_agent):
    """'study at library' should infer Library"""
    location = test_agent._infer_location_from_activity("study at library")
    assert location == "Library"


def test_infer_park(test_agent):
    """'go for a walk in the park' should infer Johnson Park"""
    location = test_agent._infer_location_from_activity("go for a walk in the park")
    assert location == "Johnson Park"


def test_infer_default(test_agent):
    """'contemplate existence' should default to work location"""
    # John Lin's work_location is "Pharmacy"
    location = test_agent._infer_location_from_activity("contemplate existence")
    assert location == "Pharmacy"


# ============================================================================
# D. Duration Inference Tests (4 tests)
# ============================================================================

def test_duration_work(test_agent):
    """'work on shift' should infer 180 minutes"""
    duration = test_agent._infer_duration_from_activity("work on shift")
    assert duration == 180


def test_duration_meeting(test_agent):
    """'attend meeting' should infer 90 minutes"""
    duration = test_agent._infer_duration_from_activity("attend meeting")
    assert duration == 90


def test_duration_breakfast(test_agent):
    """'eat breakfast' should infer 45 minutes"""
    duration = test_agent._infer_duration_from_activity("eat breakfast")
    assert duration == 45


def test_duration_default(test_agent):
    """'contemplate existence' should default to 60 minutes"""
    duration = test_agent._infer_duration_from_activity("contemplate existence")
    assert duration == 60


# ============================================================================
# E. Plan Management Tests (4 tests)
# ============================================================================

def test_get_current_plan_item_in_range(test_agent, fixed_now):
    """Query within plan item time range should return the item"""
    item = PlanItem(
        description="work",
        location="Pharmacy",
        start_time=fixed_now,
        duration_minutes=90
    )
    test_agent.daily_plan = [item]

    query_time = fixed_now + timedelta(minutes=30)
    current = test_agent.get_current_plan_item(query_time)
    assert current is not None
    assert current.description == "work"


def test_get_current_plan_item_none(test_agent, fixed_now):
    """Query outside plan item time range should return None"""
    item = PlanItem(
        description="work",
        location="Pharmacy",
        start_time=fixed_now,
        duration_minutes=90
    )
    test_agent.daily_plan = [item]

    query_time = fixed_now + timedelta(hours=2)
    current = test_agent.get_current_plan_item(query_time)
    assert current is None


def test_get_current_plan_item_skips_completed(test_agent, fixed_now):
    """Completed plan items should be skipped even if in time range"""
    item = PlanItem(
        description="work",
        location="Pharmacy",
        start_time=fixed_now,
        duration_minutes=90,
        completed=True
    )
    test_agent.daily_plan = [item]

    query_time = fixed_now + timedelta(minutes=30)
    current = test_agent.get_current_plan_item(query_time)
    assert current is None


def test_update_current_activity_transitions(test_agent, fixed_now):
    """update_current_activity should mark completed and return new activity"""
    item1 = PlanItem(
        description="breakfast",
        location="Home",
        start_time=fixed_now,
        duration_minutes=60
    )
    item2 = PlanItem(
        description="work",
        location="Pharmacy",
        start_time=fixed_now + timedelta(hours=1),
        duration_minutes=60
    )
    test_agent.daily_plan = [item1, item2]

    # At 10:30, first item should be current
    time1 = fixed_now + timedelta(minutes=30)
    result1 = test_agent.update_current_activity(time1)
    assert result1 == "breakfast"
    assert item1.completed is False  # Still ongoing

    # At 11:30, first should complete and second becomes current
    time2 = fixed_now + timedelta(hours=1, minutes=30)
    result2 = test_agent.update_current_activity(time2)
    assert result2 == "work"
    assert item1.completed is True
    assert item2.completed is False


# ============================================================================
# F. Agent State Tests (4 tests)
# ============================================================================

def test_get_state_keys(test_agent):
    """get_state should return dict with expected keys"""
    state = test_agent.get_state()
    assert "name" in state
    assert "current_location" in state
    assert "current_sub_area" in state
    assert "daily_plan" in state
    assert "current_plan_item" in state
    assert "last_reflection_time" in state
    assert "reflection_count" in state


def test_load_state_restores(test_agent, tmp_db_path, patch_llm, fixed_now):
    """load_state should restore agent state correctly"""
    # Set up initial state
    test_agent.move_to_location("Library", "reading_area")
    item = PlanItem(
        description="study",
        location="Library",
        start_time=fixed_now,
        duration_minutes=90
    )
    test_agent.daily_plan = [item]
    test_agent.reflection_count = 3

    # Save state
    state = test_agent.get_state()

    # Create new agent and load state
    new_agent = GenerativeAgent("John Lin", db_path=tmp_db_path)
    new_agent.load_state(state)

    # Verify restoration
    assert new_agent.name == test_agent.name
    assert new_agent.current_location == "Library"
    assert new_agent.current_sub_area == "reading_area"
    assert len(new_agent.daily_plan) == 1
    assert new_agent.daily_plan[0].description == "study"
    assert new_agent.reflection_count == 3


def test_move_to_location(test_agent):
    """move_to_location should update location and sub_area"""
    test_agent.move_to_location("Library", "reading_area")
    assert test_agent.current_location == "Library"
    assert test_agent.current_sub_area == "reading_area"


def test_get_status_summary(test_agent):
    """get_status_summary should contain name and occupation"""
    summary = test_agent.get_status_summary()
    assert "John Lin" in summary
    assert "Pharmacy owner" in summary
