"""Planning utilities for parsing and decomposing agent daily plans.

This module provides utilities for:
- Parsing LLM-generated daily plan text into structured PlanItem objects
- Extracting time from natural language
- Inferring locations and durations from activity descriptions
- Decomposing broad activities into specific actions
"""
import re
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from llm import get_llm_client
from prompts import DAILY_PLANNING_PROMPT

logger = logging.getLogger(__name__)

# Location inference mappings
LOCATION_KEYWORDS = {
    "home": ["wake up", "shower", "breakfast", "sleep", "bed"],
    "work": ["work", "open", "customers", "teach", "class"],
    "pharmacy": ["pharmacy", "medicine"],
    "cafe": ["cafe", "coffee", "hobbs"],
    "library": ["library", "books", "study"],
    "park": ["park", "walk", "exercise"],
    "pub": ["pub", "drink"],
    "store": ["store", "hardware", "supplies"],
    "town hall": ["town hall", "meeting", "mayor"],
}

# Default locations for home/work
DEFAULT_HOME_LOCATIONS = {
    "default": "Lin Family Home"
}

DEFAULT_WORK_LOCATIONS = {
    "default": "Oak Hill College",
    "default": "Town Center"
}


class PlanParser:
    """Parser for converting LLM-generated plans into structured PlanItems."""

    def __init__(self, default_home: str = "Lin Family Home",
                 default_work: str = "Oak Hill College"):
        """Initialize the plan parser.

        Args:
            default_home: Default location for home activities
            default_work: Default location for work activities
        """
        self.default_home = default_home
        self.default_work = default_work

    def parse(self, plan_text: str, date: datetime) -> List:
        """Parse LLM-generated daily plan into PlanItem objects.

        Args:
            plan_text: Raw text from LLM planning response
            date: The date being planned

        Returns:
            List of PlanItem objects
        """
        # Import here to avoid circular imports
        from agent import PlanItem

        plan_items = []
        lines = plan_text.split('\n')
        current_time = date.replace(hour=6, minute=0, second=0, microsecond=0)

        for line in lines:
            line = line.strip()
            if not line or not any(char.isalpha() for char in line):
                continue

            # Remove numbering and clean up
            if line.startswith(('1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)')):
                line = line[2:].strip()
            elif line.startswith(('-', '•', '*')):
                line = line[1:].strip()

            if len(line) < 10:  # Skip very short descriptions
                continue

            # Extract time if mentioned
            time_mentioned = extract_time_from_text(line)
            if time_mentioned:
                current_time = date.replace(
                    hour=time_mentioned[0],
                    minute=time_mentioned[1],
                    second=0,
                    microsecond=0
                )

            # Determine location and duration
            location = self.infer_location(line)
            duration = infer_duration(line)

            plan_item = PlanItem(
                description=line,
                location=location,
                start_time=current_time,
                duration_minutes=duration
            )

            plan_items.append(plan_item)
            current_time += timedelta(minutes=duration + 15)  # Add buffer

        return plan_items

    def infer_location(self, activity: str) -> str:
        """Infer location based on activity description.

        Args:
            activity: Activity description text

        Returns:
            Inferred location name
        """
        activity_lower = activity.lower()

        # Check each location category
        for location, keywords in LOCATION_KEYWORDS.items():
            if any(word in activity_lower for word in keywords):
                if location == "home":
                    return self.default_home
                elif location == "work":
                    return self.default_work
                # Capitalize other locations
                return location.title() if location != location.lower() else location.capitalize()

        # Default to work
        return self.default_work


def extract_time_from_text(text: str) -> Optional[Tuple[int, int]]:
    """Extract time from text (e.g., '8:30 am', '2 pm').

    Args:
        text: Text that may contain a time reference

    Returns:
        Tuple of (hour, minute) or None if no time found
    """
    # Pattern for time like "8:30 am", "2 pm", "10:00"
    time_patterns = [
        r'(\d{1,2}):(\d{2})\s*([ap]m)?',
        r'(\d{1,2})\s*([ap]m)',
        r'at\s*(\d{1,2}):(\d{2})',
        r'at\s*(\d{1,2})\s*([ap]m)'
    ]

    text_lower = text.lower()
    for pattern in time_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                hour = int(match.group(1))
                minute = int(match.group(2)) if len(match.groups()) > 1 and match.group(2) else 0
                am_pm = match.group(-1) if len(match.groups()) > 2 else None

                if am_pm == 'pm' and hour != 12:
                    hour += 12
                elif am_pm == 'am' and hour == 12:
                    hour = 0

                return (hour, minute)
            except (ValueError, IndexError):
                continue

    return None


def infer_duration(activity: str) -> int:
    """Infer activity duration in minutes based on activity type.

    Args:
        activity: Activity description text

    Returns:
        Duration in minutes
    """
    activity_lower = activity.lower()

    # Long activities (2-4 hours)
    if any(word in activity_lower for word in ['work', 'teach', 'class', 'shift']):
        return 180  # 3 hours

    # Medium activities (1-2 hours)
    if any(word in activity_lower for word in ['meeting', 'study', 'exercise', 'shop']):
        return 90   # 1.5 hours

    # Short activities (30-60 minutes)
    if any(word in activity_lower for word in ['breakfast', 'lunch', 'dinner', 'shower', 'walk']):
        return 45   # 45 minutes

    # Very short activities (15-30 minutes)
    if any(word in activity_lower for word in ['wake up', 'check', 'quick', 'brief']):
        return 20   # 20 minutes

    # Default
    return 60  # 1 hour


async def decompose_plan_item(agent_name: str, plan_item, duration_minutes: int) -> List:
    """Decompose a broad plan item into specific actions.

    Args:
        agent_name: Name of the agent
        plan_item: PlanItem description to decompose
        duration_minutes: Total duration in minutes

    Returns:
        List of sub-PlanItem objects
    """
    from agent import PlanItem

    if duration_minutes <= 30:
        return [plan_item]

    try:
        llm = await get_llm_client()
        actions = await llm.decompose_plan_item(
            agent_name, plan_item.description, duration_minutes
        )

        if not actions:
            return [plan_item]

        # Create sub-plan items
        sub_items = []
        current_time = plan_item.start_time
        time_per_action = duration_minutes // len(actions)
        time_per_action = max(5, min(15, time_per_action))  # Clamp to 5-15 minutes

        for action in actions:
            if len(action.strip()) > 5:  # Skip very short actions
                sub_item = PlanItem(
                    description=action.strip(),
                    location=plan_item.location,
                    start_time=current_time,
                    duration_minutes=time_per_action
                )
                sub_items.append(sub_item)
                current_time += timedelta(minutes=time_per_action)

        return sub_items if sub_items else [plan_item]

    except Exception as e:
        logger.error(f"Error decomposing plan item: {e}")
        return [plan_item]
