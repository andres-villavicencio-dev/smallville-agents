"""Generative Agent implementation with memory, reflection, and planning."""
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from memory import Memory, MemoryStream
from skillbank import SkillBank, distill_reflection_skill
from llm import get_llm_client
from personas import get_agent_persona, format_agent_description
from config import IMPORTANCE_THRESHOLD, MAX_RECENT_MEMORIES, ACTION_DURATION_RANGE
import config as cfg
from prompts import (
    DAILY_PLANNING_PROMPT,
    IMPORTANCE_SCORING_PROMPT,
    REFLECTION_QUESTIONS_PROMPT,
    REFLECTION_GENERATION_PROMPT
)
from reflection_engine import ReflectionEngine, PlanningEngine

logger = logging.getLogger(__name__)

@dataclass
class PlanItem:
    """A planned activity for an agent."""
    description: str
    location: str
    start_time: datetime
    duration_minutes: int
    completed: bool = False
    
    def end_time(self) -> datetime:
        return self.start_time + timedelta(minutes=self.duration_minutes)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "location": self.location,
            "start_time": self.start_time.isoformat(),
            "duration_minutes": self.duration_minutes,
            "completed": self.completed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanItem':
        return cls(
            description=data["description"],
            location=data["location"],
            start_time=datetime.fromisoformat(data["start_time"]),
            duration_minutes=data["duration_minutes"],
            completed=data.get("completed", False)
        )

class GenerativeAgent:
    """A generative agent with memory, reflection, and planning capabilities."""
    
    def __init__(self, name: str, db_path: str = "db/memories.db"):
        self.name = name
        self.persona = get_agent_persona(name)
        self.memory_stream = MemoryStream(name, db_path)
        self.skill_bank = SkillBank(name, db_path)
        self.daily_plan: List[PlanItem] = []
        self.current_plan_item: Optional[PlanItem] = None
        self.current_location = ""
        self.current_sub_area = ""
        self.last_reflection_time = datetime.now()
        self.reflection_count = 0
        
        # Initialize with special memories if defined
        self._initialize_special_memories()
    
    def _initialize_special_memories(self):
        """Initialize agent with any special memories from persona."""
        special_memory = self.persona.get("special_memory")
        if special_memory:
            memory = Memory(
                agent_name=self.name,
                description=special_memory,
                memory_type="plan",
                importance_score=8,
                location=self.persona.get("work_location", "")
            )
            self.memory_stream.add_memory(memory)
            logger.info(f"Added special memory for {self.name}: {special_memory}")
    
    async def observe(self, observation: str, location: str = "") -> Memory:
        """Process an observation and add it to memory with importance scoring."""
        try:
            # Score importance using LLM (single-model even in committee mode — fast/cheap)
            llm = await get_llm_client()
            importance_score = await llm.score_importance(observation, agent_name=self.name)
            
            # Create and store memory
            memory = Memory(
                agent_name=self.name,
                description=observation,
                memory_type="observation",
                importance_score=importance_score,
                location=location or self.current_location
            )
            
            memory_id = self.memory_stream.add_memory(memory)
            logger.debug(f"{self.name} observed (importance {importance_score}): {observation}")
            
            # Check if reflection should be triggered
            await self._check_reflection_trigger()
            
            return memory
        
        except Exception as e:
            logger.error(f"Error processing observation for {self.name}: {e}")
            # Create a basic memory without LLM scoring
            memory = Memory(
                agent_name=self.name,
                description=observation,
                memory_type="observation",
                importance_score=5,  # Default moderate importance
                location=location or self.current_location
            )
            self.memory_stream.add_memory(memory)
            return memory
    
    async def _check_reflection_trigger(self):
        """Check if reflection should be triggered based on recent importance."""
        # Cooldown: don't reflect if we reflected in the last 5 minutes (real time)
        if (datetime.now() - self.last_reflection_time).total_seconds() < 300:
            return
        
        # Only count importance of memories created SINCE last reflection
        # This effectively "resets" the accumulator after each reflection
        recent_importance_sum = self.memory_stream.get_importance_since(
            since=self.last_reflection_time, exclude_types=["reflection"]
        )
        
        if recent_importance_sum >= IMPORTANCE_THRESHOLD:
            logger.info(f"{self.name} reflection triggered (importance sum: {recent_importance_sum})")
            # Always update last_reflection_time BEFORE reflecting
            # This prevents re-triggering if the reflection fails
            self.last_reflection_time = datetime.now()
            await self.reflect()
    
    async def reflect(self) -> List[Memory]:
        """Perform reflection process as described in the paper.

        Reflection is the agent's ability to synthesize recent observations into
        higher-level insights. This method:

        1. Checks if enough recent memories have accumulated to warrant reflection
        2. Uses the appropriate strategy (single-model or committee) to generate
           reflection questions and answers based on recent memories
        3. Stores the reflection as a high-importance memory

        Returns:
            List of Memory objects created from reflection (usually 1-3)
        """
        try:
            # Get recent memories
            recent_memories = self.memory_stream.get_memories(limit=MAX_RECENT_MEMORIES)
            if len(recent_memories) < 3:
                return []

            # Use reflection strategy (handles committee vs single-model)
            reflection_strategy = ReflectionEngine.get_engine()
            reflections = await reflection_strategy.reflect(self)

            # Post-process: update reflection count and trigger skill distillation
            if reflections:
                self.last_reflection_time = datetime.now()
                self.reflection_count += 1

                # Distill skill from reflection (fire-and-forget, don't block)
                for reflection in reflections:
                    asyncio.create_task(
                        self._distill_skill_from_reflection(reflection.description)
                    )

            return reflections

        except Exception as e:
            logger.error(f"Error during reflection for {self.name}: {e}")
            return []
    
    async def plan_daily_schedule(self, date: datetime) -> List[PlanItem]:
        """Generate a daily plan using the agent's persona and memories.

        Uses the planning engine (single-model or committee) to generate a daily
        schedule, then parses and decomposes the plan into actionable items.

        Args:
            date: The date to plan for

        Returns:
            List of PlanItem objects representing the day's activities
        """
        try:
            # Get agent description and recent memories for context
            agent_description = format_agent_description(self.name)
            date_str = date.strftime("%A, %B %d, %Y")

            # Use planning engine (handles committee vs single-model)
            planning_engine = PlanningEngine.get_engine()
            response = await planning_engine.plan_day(self, date)
            
            # Parse the response into plan items
            plan_items = await self._parse_daily_plan(response, date)
            
            # Decompose broad activities into specific actions
            detailed_plan = []
            for item in plan_items:
                if item.duration_minutes > 30:  # Decompose longer activities
                    sub_actions = await self._decompose_plan_item(item)
                    detailed_plan.extend(sub_actions)
                else:
                    detailed_plan.append(item)
            
            self.daily_plan = detailed_plan
            
            # Store plan in memory
            plan_description = f"My plan for {date_str}: " + "; ".join([
                f"{item.description} at {item.start_time.strftime('%H:%M')}"
                for item in detailed_plan[:5]  # First 5 items
            ])
            
            plan_memory = Memory(
                agent_name=self.name,
                description=plan_description,
                memory_type="plan",
                importance_score=7,
                location=self.current_location
            )
            self.memory_stream.add_memory(plan_memory)
            
            logger.info(f"{self.name} planned day with {len(detailed_plan)} activities")
            return detailed_plan
        
        except Exception as e:
            logger.error(f"Error planning daily schedule for {self.name}: {e}")
            return []
    
    async def _parse_daily_plan(self, plan_text: str, date: datetime) -> List[PlanItem]:
        """Parse LLM-generated daily plan text into PlanItem objects.

        Parses lines from the LLM response, extracting times, inferring locations
        and durations, and creating PlanItem objects.

        Args:
            plan_text: Raw text from LLM daily plan
            date: The date being planned for

        Returns:
            List of PlanItem objects with parsed data
        """
        plan_items = []
        lines = plan_text.split('\n')
        current_time = date.replace(hour=6, minute=0, second=0, microsecond=0)  # Start at 6 AM
        
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
            time_mentioned = self._extract_time_from_text(line)
            if time_mentioned:
                current_time = date.replace(
                    hour=time_mentioned[0],
                    minute=time_mentioned[1],
                    second=0,
                    microsecond=0
                )
            
            # Determine location based on activity description
            location = self._infer_location_from_activity(line)
            
            # Determine duration based on activity type
            duration = self._infer_duration_from_activity(line)
            
            plan_item = PlanItem(
                description=line,
                location=location,
                start_time=current_time,
                duration_minutes=duration
            )
            
            plan_items.append(plan_item)
            current_time += timedelta(minutes=duration + 15)  # Add buffer time
        
        return plan_items
    
    def _extract_time_from_text(self, text: str) -> Optional[Tuple[int, int]]:
        """Extract time from text (e.g., '8:30 am', '2 pm')."""
        import re
        
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
    
    def _infer_location_from_activity(self, activity: str) -> str:
        """Infer location based on activity description."""
        activity_lower = activity.lower()
        
        # Home activities
        if any(word in activity_lower for word in ['wake up', 'shower', 'breakfast', 'sleep', 'bed']):
            return self.persona.get("home_location", "Lin Family Home")
        
        # Work activities
        if any(word in activity_lower for word in ['work', 'open', 'customers', 'teach', 'class']):
            return self.persona.get("work_location", "Oak Hill College")
        
        # Specific locations
        if any(word in activity_lower for word in ['pharmacy', 'medicine']):
            return "Pharmacy"
        elif any(word in activity_lower for word in ['cafe', 'coffee', 'hobbs']):
            return "Hobbs Cafe"
        elif any(word in activity_lower for word in ['library', 'books', 'study']):
            return "Library"
        elif any(word in activity_lower for word in ['park', 'walk', 'exercise']):
            return "Johnson Park"
        elif any(word in activity_lower for word in ['pub', 'drink']):
            return "The Rose and Crown Pub"
        elif any(word in activity_lower for word in ['store', 'hardware', 'supplies']):
            return "Harvey Oak Supply Store"
        elif any(word in activity_lower for word in ['town hall', 'meeting', 'mayor']):
            return "Town Hall"
        
        # Default to work location
        return self.persona.get("work_location", self.current_location or "Town Center")
    
    def _infer_duration_from_activity(self, activity: str) -> int:
        """Infer activity duration in minutes."""
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
    
    async def _decompose_plan_item(self, plan_item: PlanItem) -> List[PlanItem]:
        """Decompose a broad plan item into specific actions."""
        if plan_item.duration_minutes <= 30:
            return [plan_item]
        
        try:
            llm = await get_llm_client()
            actions = await llm.decompose_plan_item(
                self.name, plan_item.description, plan_item.duration_minutes
            )
            
            if not actions:
                return [plan_item]
            
            # Create sub-plan items
            sub_items = []
            current_time = plan_item.start_time
            time_per_action = plan_item.duration_minutes // len(actions)
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
    
    def get_current_plan_item(self, current_time: datetime) -> Optional[PlanItem]:
        """Get the plan item that should be active at the current time."""
        for item in self.daily_plan:
            if (item.start_time <= current_time <= item.end_time() and 
                not item.completed):
                return item
        return None
    
    def update_current_activity(self, current_time: datetime) -> Optional[str]:
        """Update current activity based on the plan and return activity description."""
        current_item = self.get_current_plan_item(current_time)
        
        if current_item != self.current_plan_item:
            if self.current_plan_item:
                self.current_plan_item.completed = True
                logger.debug(f"{self.name} completed: {self.current_plan_item.description}")
            
            self.current_plan_item = current_item
            if current_item:
                logger.debug(f"{self.name} started: {current_item.description}")
                return current_item.description
        
        return None
    
    async def _distill_skill_from_reflection(self, reflection_text: str):
        """Background task: distill a skill from a reflection."""
        try:
            context = f"at {self.current_location}, activity: {self.current_plan_item.description if self.current_plan_item else 'idle'}"
            skill = await distill_reflection_skill(self.name, reflection_text, context)
            if skill:
                self.skill_bank.add_skill(skill)
        except Exception as e:
            logger.debug(f"Skill distillation failed (non-critical): {e}")

    def get_relevant_skills_text(self, context: str) -> str:
        """Retrieve relevant skills formatted for LLM prompts."""
        skills = self.skill_bank.retrieve_relevant_skills(context, top_k=3)
        return self.skill_bank.format_skills_for_prompt(skills)

    async def react_to_conversation(self, other_agent: str, conversation_summary: str, 
                                      current_time: datetime) -> bool:
        """React to a completed conversation by potentially modifying the daily plan.
        Returns True if the plan was modified."""
        logger.info(f"[replan] {self.name} evaluating plan after talking to {other_agent}")
        # Cap re-plans to avoid churn
        if not hasattr(self, '_replan_count'):
            self._replan_count = 0
        if self._replan_count >= 3:
            logger.info(f"[replan] {self.name} hit re-plan cap, skipping")
            return False
        
        # Get remaining plan items
        remaining = [item for item in self.daily_plan 
                     if item.end_time() > current_time and not item.completed]
        if not remaining:
            remaining_text = "No remaining plans for today."
        else:
            remaining_text = "\n".join(
                f"- {item.start_time.strftime('%H:%M')} — {item.description} at {item.location}"
                for item in remaining[:8]
            )
        
        # Truncate conversation summary
        conv_short = conversation_summary[:500] if len(conversation_summary) > 500 else conversation_summary
        
        prompt = (
            f"You are {self.name}. You just finished talking to {other_agent}.\n\n"
            f"Conversation:\n{conv_short}\n\n"
            f"Your remaining plans today:\n{remaining_text}\n\n"
            f"Based on this conversation, should you change your plans?\n"
            f"If YES, write the new activity in this exact format:\n"
            f"  TIME - ACTIVITY at LOCATION\n"
            f"  Example: 17:00 - Attend Valentine's Day party at Hobbs Cafe\n"
            f"If NO changes needed, write: NO_CHANGE"
        )
        
        try:
            if cfg.USE_COMMITTEE:
                from committee import get_committee, EXPERTS
                committee = get_committee()
                expert = EXPERTS["social"]
                from llm import _notify_llm_status
                _notify_llm_status(self.name, "react_to_conv", expert.model)
                response = await committee._call_model(expert, prompt)
            else:
                llm = await get_llm_client()
                response = await llm.generate(prompt, temperature=0.6, max_tokens=80, task="react")
            
            if not response:
                logger.info(f"[replan] {self.name}: no change after talking to {other_agent}")
                return False
            
            # Try to parse — if it doesn't match TIME format, treat as no-change
            new_item = self._parse_replan_response(response, current_time)
            if not new_item:
                logger.info(f"[replan] {self.name}: no change — {response[:80]}")
                return False
            
            # Insert into plan: remove conflicting items, add new one
            self._inject_plan_item(new_item)
            self._replan_count += 1
            
            # Store memory about the decision
            memory = Memory(
                agent_name=self.name,
                description=f"Decided to {new_item.description} after talking to {other_agent}",
                memory_type="plan",
                importance_score=7,
                location=self.current_location
            )
            self.memory_stream.add_memory(memory)
            
            logger.info(f"[replan] {self.name} added: {new_item.start_time.strftime('%H:%M')} — {new_item.description} at {new_item.location}")
            return True
            
        except Exception as e:
            logger.error(f"Error in react_to_conversation for {self.name}: {e}")
            return False
    
    def _parse_replan_response(self, response: str, current_time: datetime) -> Optional[PlanItem]:
        """Parse a replan response like '17:00 - Attend party at Hobbs Cafe'."""
        import re
        # Match pattern: HH:MM - description at location
        m = re.search(r'(\d{1,2}):(\d{2})\s*[-–]\s*(.+?)\s+at\s+(.+)', response, re.IGNORECASE)
        if not m:
            return None
        
        hour, minute = int(m.group(1)), int(m.group(2))
        description = m.group(3).strip()
        location = m.group(4).strip().rstrip('.')
        
        # Build start time using current_time's date
        start_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Only accept future activities
        if start_time <= current_time:
            return None
        
        return PlanItem(
            description=description,
            location=location,
            start_time=start_time,
            duration_minutes=self._infer_duration_from_activity(description)
        )
    
    def _inject_plan_item(self, new_item: PlanItem):
        """Insert a new plan item, removing or shortening conflicts."""
        new_start = new_item.start_time
        new_end = new_item.end_time()
        
        updated_plan = []
        for item in self.daily_plan:
            # Keep items that don't overlap
            if item.end_time() <= new_start or item.start_time >= new_end:
                updated_plan.append(item)
            # Shorten items that partially overlap
            elif item.start_time < new_start and item.end_time() > new_start:
                item.duration_minutes = int((new_start - item.start_time).total_seconds() / 60)
                if item.duration_minutes > 5:
                    updated_plan.append(item)
            # Items fully inside the new item's window get dropped
        
        updated_plan.append(new_item)
        updated_plan.sort(key=lambda x: x.start_time)
        self.daily_plan = updated_plan

    def get_status_summary(self) -> str:
        """Get a summary of the agent's current status."""
        status = f"{self.name} ({self.persona.get('occupation', 'Unknown')})"
        if self.current_location:
            status += f" at {self.current_location}"
        if self.current_plan_item:
            status += f" - {self.current_plan_item.description}"
        return status
    
    def move_to_location(self, location: str, sub_area: str = ""):
        """Update agent's location."""
        if location != self.current_location:
            logger.debug(f"{self.name} moved from {self.current_location} to {location}")
        self.current_location = location
        self.current_sub_area = sub_area
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state for saving/loading."""
        return {
            "name": self.name,
            "current_location": self.current_location,
            "current_sub_area": self.current_sub_area,
            "daily_plan": [item.to_dict() for item in self.daily_plan],
            "current_plan_item": self.current_plan_item.to_dict() if self.current_plan_item else None,
            "last_reflection_time": self.last_reflection_time.isoformat(),
            "reflection_count": self.reflection_count
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load agent state."""
        self.current_location = state.get("current_location", "")
        self.current_sub_area = state.get("current_sub_area", "")
        
        # Load daily plan
        plan_data = state.get("daily_plan", [])
        self.daily_plan = [PlanItem.from_dict(item) for item in plan_data]
        
        # Load current plan item
        current_item_data = state.get("current_plan_item")
        if current_item_data:
            self.current_plan_item = PlanItem.from_dict(current_item_data)
        
        # Load reflection data
        if "last_reflection_time" in state:
            self.last_reflection_time = datetime.fromisoformat(state["last_reflection_time"])
        self.reflection_count = state.get("reflection_count", 0)