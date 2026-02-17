"""Reflection engine with strategy pattern for committee vs single-model modes.

This module provides a unified interface for reflection operations, abstracting away
the differences between single-model and committee-of-experts modes.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from memory import Memory
from llm import get_llm_client
from config import IMPORTANCE_THRESHOLD, MAX_RECENT_MEMORIES, IMPORTANCE_THRESHOLD
import config as cfg

if TYPE_CHECKING:
    from agent import GenerativeAgent

logger = logging.getLogger(__name__)


class ReflectionStrategy(ABC):
    """Abstract base class for reflection strategies."""

    @abstractmethod
    async def reflect(self, agent: 'GenerativeAgent') -> List[Memory]:
        """Perform reflection and return reflection memories.

        Args:
            agent: The agent performing reflection

        Returns:
            List of Memory objects created from reflection
        """
        pass


class SingleModelReflection(ReflectionStrategy):
    """Reflection using a single LLM model."""

    async def reflect(self, agent: 'GenerativeAgent') -> List[Memory]:
        """Perform reflection using single model approach.

        Steps:
        1. Get recent memories
        2. Generate reflection questions from memories
        3. For each question, retrieve relevant memories and generate reflection
        4. Store reflection as memory
        """
        recent_memories = agent.memory_stream.get_memories(limit=MAX_RECENT_MEMORIES)
        if len(recent_memories) < 3:
            return []

        memory_descriptions = [mem.description for mem in recent_memories]
        llm = await get_llm_client()

        # Step 1: Generate reflection questions
        questions = await llm.generate_reflection_questions(
            memory_descriptions, agent_name=agent.name
        )
        if not questions:
            logger.warning(f"No reflection questions generated for {agent.name}")
            return []

        reflections = []

        # Step 2: For each question, retrieve relevant memories and generate reflection
        for question in questions:
            try:
                relevant_memories = agent.memory_stream.retrieve_memories(question, top_k=10)
                relevant_descriptions = [mem[0].description for mem in relevant_memories]
                source_memory_ids = [mem[0].id for mem in relevant_memories if mem[0].id]

                if not relevant_descriptions:
                    continue

                # Generate reflection
                reflection_text = await llm.generate_reflection(
                    question, relevant_descriptions, agent_name=agent.name
                )

                if reflection_text:
                    reflection_memory = Memory(
                        agent_name=agent.name,
                        description=reflection_text,
                        memory_type="reflection",
                        importance_score=8,
                        location=agent.current_location,
                        source_memory_ids=source_memory_ids
                    )

                    memory_id = agent.memory_stream.add_memory(reflection_memory)
                    reflections.append(reflection_memory)

                    logger.info(f"{agent.name} reflected: {reflection_text}")

            except Exception as e:
                logger.error(f"Error generating reflection for question '{question}': {e}")
                continue

        return reflections


class CommitteeReflection(ReflectionStrategy):
    """Reflection using committee of expert models."""

    async def reflect(self, agent: 'GenerativeAgent') -> List[Memory]:
        """Perform reflection using committee approach.

        Uses Memory + Emotional + Judge pipeline for synthesis.
        """
        from committee import reflect as committee_reflect

        recent_memories = agent.memory_stream.get_memories(limit=MAX_RECENT_MEMORIES)
        if len(recent_memories) < 3:
            return []

        memory_descriptions = [mem.description for mem in recent_memories]
        memories_text = "\n".join(f"- {m}" for m in memory_descriptions[:30])

        situation = (
            f"{agent.name} is at {agent.current_location}. "
            f"Current activity: {agent.current_plan_item.description if agent.current_plan_item else 'idle'}."
        )

        reflection_text = await committee_reflect(
            agent.name, situation, memories=memories_text
        )

        if reflection_text:
            reflection_memory = Memory(
                agent_name=agent.name,
                description=reflection_text,
                memory_type="reflection",
                importance_score=8,
                location=agent.current_location,
            )
            agent.memory_stream.add_memory(reflection_memory)

            logger.info(f"{agent.name} reflected (committee): {reflection_text}")
            return [reflection_memory]

        return []


class ReflectionEngine:
    """Factory for getting the appropriate reflection strategy."""

    _instance: Optional[ReflectionStrategy] = None

    @classmethod
    def get_engine(cls, use_committee: bool = None) -> ReflectionStrategy:
        """Get the reflection strategy based on configuration.

        Args:
            use_committee: Override committee mode. If None, uses config.USE_COMMITTEE

        Returns:
            ReflectionStrategy instance (single-model or committee)
        """
        if use_committee is None:
            use_committee = cfg.USE_COMMITTEE

        if use_committee:
            return CommitteeReflection()
        else:
            return SingleModelReflection()


class PlanningStrategy(ABC):
    """Abstract base class for planning strategies."""

    @abstractmethod
    async def plan_day(self, agent: 'GenerativeAgent', date) -> str:
        """Generate daily plan text.

        Args:
            agent: The agent to plan for
            date: The date to plan for

        Returns:
            Raw plan text from LLM
        """
        pass


class SingleModelPlanning(PlanningStrategy):
    """Planning using a single LLM model."""

    async def plan_day(self, agent: 'GenerativeAgent', date) -> str:
        """Generate daily plan using single model."""
        from personas import format_agent_description, get_agent_persona

        # Fix 2: Use retrieve_memories() with focal points instead of get_memories()
        events_query = f"What events or social gatherings are happening? {agent.name}"
        plans_query = f"What are {agent.name}'s plans and commitments?"
        
        events_memories = agent.memory_stream.retrieve_memories(events_query, top_k=10)
        plans_memories = agent.memory_stream.retrieve_memories(plans_query, top_k=10)
        
        # Merge and deduplicate relevant memories
        all_relevant = list(events_memories) + list(plans_memories)
        seen_ids = set()
        unique_memories = []
        for mem, score in all_relevant:
            if mem.id not in seen_ids:
                unique_memories.append(mem)
                seen_ids.add(mem.id)
        
        memories_text = "\n".join(f"- {m.description}" for m in unique_memories[:15])

        llm = await get_llm_client()
        agent_description = format_agent_description(agent.name)
        date_str = date.strftime("%A, %B %d, %Y")

        persona = get_agent_persona(agent.name)
        home = persona.get("home_location", "Lin Family Home")
        work = persona.get("work_location", "Oak Hill College")
        lunch_location = persona.get("lunch_location", "Hobbs Cafe")
        errand_locations = persona.get("errand_locations", ["Library", "Johnson Park", "Harvey Oak Supply Store"])

        # Fix C: Extract event commitments from memories
        fixed_commitments = CommitteePlanning._extract_event_commitments(unique_memories, agent.name)
        commitments_block = ""
        if fixed_commitments:
            commitments_block = (
                "\nFIXED COMMITMENTS (these MUST appear in the schedule at the specified times):\n"
                + "\n".join(f"- {c}" for c in fixed_commitments)
                + "\nPlan other activities AROUND these fixed events.\n"
            )

        # Fix B: Full-day prompt structure
        enhanced_context = (
            f"Relevant memories and events:\n{memories_text}\n"
            f"{commitments_block}\n"
            f"Home: {home}. Work: {work}.\n"
            f"Suggested lunch location: {lunch_location}.\n"
            f"Suggested errand locations: {', '.join(errand_locations)}.\n"
            f"Generate a FULL daily schedule covering 6:00 AM to 10:00 PM.\n"
            f"MUST include MORNING (6-12), AFTERNOON (12-5), and EVENING (5-10) activities."
        )

        return await llm.generate_daily_plan(
            agent.name, agent_description, date_str, context=enhanced_context
        )


class CommitteePlanning(PlanningStrategy):
    """Planning using committee of expert models."""

    async def plan_day(self, agent: 'GenerativeAgent', date) -> str:
        """Generate daily plan using committee approach.

        Uses Temporal + Memory + Spatial + Judge pipeline.
        """
        from committee import plan_day as committee_plan
        from personas import format_agent_description

        # Fix 2: Use retrieve_memories() with focal points instead of get_memories()
        events_query = f"What events or social gatherings are happening? {agent.name}"
        plans_query = f"What are {agent.name}'s plans and commitments?"
        
        events_memories = agent.memory_stream.retrieve_memories(events_query, top_k=10)
        plans_memories = agent.memory_stream.retrieve_memories(plans_query, top_k=10)
        
        # Merge and deduplicate relevant memories
        all_relevant = list(events_memories) + list(plans_memories)
        seen_ids = set()
        unique_memories = []
        for mem, score in all_relevant:
            if mem.id not in seen_ids:
                unique_memories.append(mem)
                seen_ids.add(mem.id)
        
        memories_text = "\n".join(f"- {m.description}" for m in unique_memories[:15])

        agent_description = format_agent_description(agent.name)
        date_str = date.strftime("%A, %B %d, %Y")

        from personas import get_agent_persona
        persona = get_agent_persona(agent.name)
        home = persona.get("home_location", "Lin Family Home")
        work = persona.get("work_location", "Oak Hill College")
        lunch_location = persona.get("lunch_location", "Hobbs Cafe")
        errand_locations = persona.get("errand_locations", ["Library", "Johnson Park", "Harvey Oak Supply Store"])

        # Fix C: Extract high-importance event memories and inject as fixed commitments
        fixed_commitments = self._extract_event_commitments(unique_memories, agent.name)
        commitments_block = ""
        if fixed_commitments:
            commitments_block = (
                "\nFIXED COMMITMENTS (these MUST appear in the schedule at the specified times):\n"
                + "\n".join(f"- {c}" for c in fixed_commitments)
                + "\nPlan other activities AROUND these fixed events.\n"
            )

        # Fix B: Restructured prompt requiring full-day coverage 6AM-10PM
        situation = (
            f"{agent_description}\n"
            f"Date: {date_str}\n"
            f"Home: {home}. Work: {work}.\n"
            f"Suggested lunch location: {lunch_location}.\n"
            f"Suggested errand locations: {', '.join(errand_locations)}.\n"
            f"Relevant memories and events:\n{memories_text}\n"
            f"{commitments_block}\n"
            f"Generate a FULL daily schedule for {agent.name} covering 6:00 AM to 10:00 PM.\n"
            f"The schedule MUST include ALL three periods:\n"
            f"  MORNING (6:00 AM - 12:00 PM): Wake up, morning routine, work/activities\n"
            f"  AFTERNOON (12:00 PM - 5:00 PM): Lunch, errands, social visits\n"
            f"  EVENING (5:00 PM - 10:00 PM): Events, dinner, leisure, return home\n\n"
            f"Output exactly 8-10 lines. Format each line as:\n"
            f"TIME AM/PM - activity at LOCATION\n"
            f"Example:\n"
            f"6:00 AM - wake up and morning routine at {home}\n"
            f"8:00 AM - open the pharmacy for the day at Pharmacy\n"
            f"12:00 PM - have lunch at Hobbs Cafe\n"
            f"5:00 PM - attend community event at Johnson Park\n"
            f"9:00 PM - wind down and prepare for bed at {home}"
        )

        return await committee_plan(
            agent.name, situation, memories=memories_text
        )

    @staticmethod
    def _extract_event_commitments(memories, agent_name: str) -> list:
        """Extract time-specific events from high-importance memories to use as schedule anchors.
        
        Looks for memories about parties, events, gatherings etc. that have specific
        times mentioned. Returns them as fixed commitment strings for the planner.
        """
        import re
        from config import SMALLVILLE_LOCATIONS
        
        commitments = []
        event_keywords = ['party', 'celebration', 'festival', 'gathering', 'event', 'ceremony', 'concert']
        # Match times like "5 PM", "5:00 PM", "5-7 PM", "5:00 PM - 7:00 PM"
        # Also handles "5-7 PM" where only the end has AM/PM
        time_pattern = re.compile(
            r'(\d{1,2}(?::\d{2})?)\s*(?:AM|PM|am|pm)?'
            r'\s*[-–to]+\s*'
            r'(\d{1,2}(?::\d{2})?)\s*(AM|PM|am|pm)'
        )
        # Simpler pattern for single times like "at 3:00 PM"
        single_time_pattern = re.compile(
            r'(\d{1,2}(?::\d{2})?)\s*(AM|PM|am|pm)'
        )
        
        known_locations = set(SMALLVILLE_LOCATIONS.keys())
        
        for mem in memories:
            desc_lower = mem.description.lower()
            # Only consider high-importance event memories
            if mem.importance_score < 6:
                continue
            if not any(kw in desc_lower for kw in event_keywords):
                continue
            
            # Try to extract a time range first (e.g. "5-7 PM"), then single time
            range_match = time_pattern.search(mem.description)
            if range_match:
                start_num = range_match.group(1).strip()
                end_num = range_match.group(2).strip()
                ampm = range_match.group(3).upper()
                time_str = f"{start_num} {ampm}"  # Use start time as the anchor
            else:
                single_match = single_time_pattern.search(mem.description)
                if not single_match:
                    continue
                time_str = f"{single_match.group(1)} {single_match.group(2).upper()}"
            
            # Extract location: match known Smallville locations
            location = ""
            for loc in known_locations:
                if loc.lower() in desc_lower:
                    location = loc
                    break
            
            # Build a clean event name from the keywords found
            event_type = next((kw for kw in event_keywords if kw in desc_lower), "event")
            # Capitalize nicely
            event_name = f"Valentine's Day {event_type}" if 'valentine' in desc_lower else event_type.title()
            
            commitment = f"{time_str} - Attend {event_name} at {location}" if location else f"{time_str} - Attend {event_name}"
            
            if commitment not in commitments:
                commitments.append(commitment)
        
        return commitments[:3]  # Max 3 fixed commitments to leave room for organic planning


class PlanningEngine:
    """Factory for getting the appropriate planning strategy."""

    @classmethod
    def get_engine(cls, use_committee: bool = None) -> PlanningStrategy:
        """Get the planning strategy based on configuration."""
        if use_committee is None:
            use_committee = cfg.USE_COMMITTEE

        if use_committee:
            return CommitteePlanning()
        else:
            return SingleModelPlanning()


class ConversationStrategy(ABC):
    """Abstract base class for conversation strategies."""

    @abstractmethod
    async def should_initiate(self, initiator: str, target: str, context: str,
                              memory_stream) -> bool:
        """Determine if conversation should start."""
        pass

    @abstractmethod
    async def generate_response(self, speaker: str, other_agent: str,
                               conversation, memory_stream) -> str:
        """Generate conversation response."""
        pass


class SingleModelConversation(ConversationStrategy):
    """Conversation using a single LLM model."""

    async def should_initiate(self, initiator: str, target: str, context: str,
                              memory_stream) -> bool:
        """Determine if should start conversation (single model)."""
        from prompts import CONVERSATION_INITIATION_PROMPT

        # Fix 4: Broaden query to include events and parties
        query = f"{target} plans events party invite {initiator} activities conversation talking"
        relevant_memories = memory_stream.retrieve_memories(query, top_k=5)
        memory_descriptions = [mem[0].description for mem in relevant_memories]

        llm = await get_llm_client()
        prompt = CONVERSATION_INITIATION_PROMPT.format(
            agent_name=initiator,
            other_agent=target,
            context=context,
            agent_memories="\n".join(memory_descriptions) if memory_descriptions
                          else "No recent relevant memories"
        )

        response = await llm.generate(prompt, temperature=0.6, max_tokens=10,
                                       task="conversation")
        return response.strip().upper() == "YES"

    async def generate_response(self, speaker: str, other_agent: str,
                               conversation, memory_stream) -> str:
        """Generate conversation response (single model)."""
        from prompts import CONVERSATION_RESPONSE_PROMPT
        from personas import format_agent_description, get_agent_persona

        # Fix 5: Broaden query to include events and parties
        query = f"{other_agent} events plans party invite activities conversation"
        relevant_memories = memory_stream.retrieve_memories(query, top_k=8)
        memory_descriptions = [mem[0].description for mem in relevant_memories]

        persona = get_agent_persona(speaker)
        personality = persona.get("personality", "friendly")

        llm = await get_llm_client()
        prompt = CONVERSATION_RESPONSE_PROMPT.format(
            agent_name=speaker,
            other_agent=other_agent,
            conversation_history=conversation.get_history_text(),
            agent_memories="\n".join(memory_descriptions[-5:]) if memory_descriptions
                          else "No relevant memories",
            agent_personality=personality
        )

        response = await llm.generate(prompt, temperature=0.9, max_tokens=100,
                                        task="conversation")
        return response.strip()


class CommitteeConversation(ConversationStrategy):
    """Conversation using committee of expert models."""

    async def should_initiate(self, initiator: str, target: str, context: str,
                              memory_stream) -> bool:
        """Determine if should start conversation (committee)."""
        from committee import should_converse as committee_should_converse

        # Fix 4: Broaden query to include events and parties
        query = f"{target} plans events party invite {initiator} activities conversation talking"
        relevant_memories = memory_stream.retrieve_memories(query, top_k=5)
        memory_descriptions = [mem[0].description for mem in relevant_memories]

        situation = (
            f"{initiator} and {target} are both at the same location. {context}\n"
            f"Recent memories: {'; '.join(memory_descriptions[:3]) if memory_descriptions else 'None'}"
        )

        should_talk = await committee_should_converse(initiator, situation)
        logger.info(f"Conversation check (committee): {initiator} -> {target}: "
                   f"{'YES' if should_talk else 'NO'}")
        return should_talk

    async def generate_response(self, speaker: str, other_agent: str,
                               conversation, memory_stream) -> str:
        """Generate conversation response (committee)."""
        from committee import generate_dialogue
        from personas import get_agent_persona

        persona = get_agent_persona(speaker)
        personality = persona.get("personality", "friendly")

        # Fix 5: Broaden query to include events and parties
        query = f"{other_agent} events plans party invite activities conversation"
        relevant_memories = memory_stream.retrieve_memories(query, top_k=8)
        memory_descriptions = [mem[0].description for mem in relevant_memories]

        situation = (
            f"{speaker} (personality: {personality}) is talking to {other_agent} "
            f"at {conversation.location}.\n"
            f"Conversation so far:\n{conversation.get_history_text()}\n"
            f"Generate {speaker}'s next line of dialogue."
        )
        memories_text = ("\n".join(f"- {m}" for m in memory_descriptions[-5:])
                        if memory_descriptions else "No relevant memories")

        response = await generate_dialogue(speaker, situation,
                                           memories=memories_text,
                                           talking_to=other_agent)
        return response.strip() if response else "I see."


class ConversationEngine:
    """Factory for getting the appropriate conversation strategy."""

    @classmethod
    def get_engine(cls, use_committee: bool = None) -> ConversationStrategy:
        """Get the conversation strategy based on configuration."""
        if use_committee is None:
            use_committee = cfg.USE_COMMITTEE

        if use_committee:
            return CommitteeConversation()
        else:
            return SingleModelConversation()
