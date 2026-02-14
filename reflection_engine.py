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
        from personas import format_agent_description

        llm = await get_llm_client()
        agent_description = format_agent_description(agent.name)
        date_str = date.strftime("%A, %B %d, %Y")

        return await llm.generate_daily_plan(
            agent.name, agent_description, date_str
        )


class CommitteePlanning(PlanningStrategy):
    """Planning using committee of expert models."""

    async def plan_day(self, agent: 'GenerativeAgent', date) -> str:
        """Generate daily plan using committee approach.

        Uses Temporal + Memory + Spatial + Judge pipeline.
        """
        from committee import plan_day as committee_plan
        from personas import format_agent_description

        recent_memories = agent.memory_stream.get_memories(limit=20)
        memories_text = "\n".join(f"- {m.description}" for m in recent_memories)

        agent_description = format_agent_description(agent.name)
        date_str = date.strftime("%A, %B %d, %Y")

        situation = (
            f"{agent_description}\n"
            f"Date: {date_str}\n"
            f"Generate a detailed daily schedule for {agent.name} with 6-8 activities, "
            f"including times and locations. Format: time - activity at location"
        )

        return await committee_plan(
            agent.name, situation, memories=memories_text
        )


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

        query = f"conversation with {target} or {target} or talking"
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

        query = f"conversation with {other_agent} or {other_agent}"
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

        query = f"conversation with {target} or {target} or talking"
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

        query = f"conversation with {other_agent} or {other_agent}"
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
                                           memories=memories_text)
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
