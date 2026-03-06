"""Conversation system for generative agents."""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config as cfg
from llm import get_llm_client
from memory import Memory, MemoryStream
from personas import format_agent_description, get_agent_persona
from skillbank import SkillBank, distill_conversation_skill

try:
    from voice_integration import voice_conversation_end
except ImportError:
    voice_conversation_end = None
from prompts import (
    CONVERSATION_ENDING_PROMPT,
    CONVERSATION_INITIATION_PROMPT,
    CONVERSATION_RESPONSE_PROMPT,
    CONVERSATION_TOPIC_PROMPT,
)
from reflection_engine import ConversationEngine

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    speaker: str
    message: str
    timestamp: datetime
    
class Conversation:
    """Manages a conversation between two agents."""
    
    def __init__(self, agent1: str, agent2: str, location: str):
        self.agent1 = agent1
        self.agent2 = agent2
        self.location = location
        self.turns: list[ConversationTurn] = []
        self.start_time = datetime.now()
        self.active = True
        self.max_turns = 8
    
    def add_turn(self, speaker: str, message: str):
        """Add a turn to the conversation."""
        turn = ConversationTurn(
            speaker=speaker,
            message=message,
            timestamp=datetime.now()
        )
        self.turns.append(turn)
        logger.info(f"Conversation: {speaker}: {message}")
    
    def get_history_text(self) -> str:
        """Get conversation history as formatted text."""
        history = []
        for turn in self.turns:
            history.append(f"{turn.speaker}: {turn.message}")
        return "\n".join(history)
    
    def should_end(self) -> bool:
        """Check if conversation should end."""
        return len(self.turns) >= self.max_turns or not self.active
    
    def get_participants(self) -> tuple[str, str]:
        """Get the two participants."""
        return self.agent1, self.agent2

class ConversationManager:
    """Manages conversations between agents."""
    
    def __init__(self):
        self.active_conversations: dict[tuple[str, str], Conversation] = {}
        self.conversation_history: list[Conversation] = []
    
    def get_conversation_key(self, agent1: str, agent2: str) -> tuple[str, str]:
        """Get a consistent key for agent pair."""
        return tuple(sorted([agent1, agent2]))
    
    def has_active_conversation(self, agent1: str, agent2: str) -> bool:
        """Check if agents have an active conversation."""
        key = self.get_conversation_key(agent1, agent2)
        return key in self.active_conversations
    
    def is_agent_busy(self, agent_name: str) -> bool:
        """Check if an agent is already in ANY active conversation."""
        for (a1, a2) in self.active_conversations:
            if agent_name in (a1, a2):
                return True
        return False
    
    @property
    def active_count(self) -> int:
        """Number of currently active conversations."""
        return len(self.active_conversations)
    
    def get_active_conversation(self, agent1: str, agent2: str) -> Conversation | None:
        """Get active conversation between agents."""
        key = self.get_conversation_key(agent1, agent2)
        return self.active_conversations.get(key)
    
    async def should_initiate_conversation(self, initiator: str, target: str,
                                         context: str, memory_stream: MemoryStream) -> bool:
        """Determine if initiator should start conversation with target.

        Uses the conversation engine (single-model or committee) to decide
        whether two agents at the same location should start talking.

        Args:
            initiator: Name of agent initiating conversation
            target: Name of agent being approached
            context: Context string (e.g., "Both at Hobbs Cafe")
            memory_stream: Initiator's memory stream for context

        Returns:
            True if conversation should start, False otherwise
        """
        try:
            # Use conversation engine (handles committee vs single-model)
            conv_engine = ConversationEngine.get_engine()
            return await conv_engine.should_initiate(initiator, target, context, memory_stream)

        except Exception as e:
            logger.error(f"Error determining conversation initiation: {e}")
            return False
    
    async def start_conversation(self, agent1: str, agent2: str, 
                               location: str, memory_streams: dict[str, MemoryStream]) -> Conversation | None:
        """Start a new conversation between two agents."""
        key = self.get_conversation_key(agent1, agent2)
        
        if key in self.active_conversations:
            logger.warning(f"Conversation already active between {agent1} and {agent2}")
            return self.active_conversations[key]
        
        try:
            # Create new conversation
            conversation = Conversation(agent1, agent2, location)
            
            # Generate opening message from agent1
            opening_message = await self._generate_opening_message(
                agent1, agent2, location, memory_streams[agent1]
            )
            
            if opening_message:
                conversation.add_turn(agent1, opening_message)
                self.active_conversations[key] = conversation
                
                # Add conversation start to both agents' memories
                await self._add_conversation_memory(
                    agent1, f"Started conversation with {agent2} at {location}",
                    memory_streams[agent1], location
                )
                await self._add_conversation_memory(
                    agent2, f"{agent1} started talking to me at {location}",
                    memory_streams[agent2], location
                )
                
                logger.info(f"Started conversation between {agent1} and {agent2} at {location}")
                return conversation
        
        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
        
        return None
    
    async def _generate_opening_message(self, speaker: str, target: str, 
                                       location: str, memory_stream: MemoryStream) -> str:
        """Generate an opening conversation message."""
        try:
            # Fix 4: Broaden query to include events and parties
            query = f"{target} plans events party invite {speaker} activities"
            relevant_memories = memory_stream.retrieve_memories(query, top_k=5)
            memory_descriptions = [mem[0].description for mem in relevant_memories[-3:]]
            
            # Get speaker's persona
            persona = get_agent_persona(speaker)
            personality = persona.get("personality", "friendly")
            
            llm = await get_llm_client()
            
            # Single LLM call for opening (merged topic + opening generation)
            memories_text = "\n".join(memory_descriptions) if memory_descriptions else "No specific memories"
            opening_prompt = f"""You are {speaker} ({personality}) starting a conversation with {target} at {location}.

Recent memories:
{memories_text}

Generate a natural, brief opening message (1-2 sentences) that {speaker} would say to {target}. Be specific and in-character."""
            
            opening = await llm.generate(opening_prompt, temperature=0.9, max_tokens=80, task="conversation")
            return opening.strip()
            
        except Exception as e:
            logger.error(f"Error generating opening message: {e}")
            return f"Hello {target}!"
    
    async def continue_conversation(self, conversation: Conversation, 
                                  memory_streams: dict[str, MemoryStream]) -> bool:
        """Continue an active conversation with the next turn."""
        if not conversation.active or conversation.should_end():
            return False
        
        try:
            # Determine who should speak next
            if not conversation.turns:
                next_speaker = conversation.agent1
            else:
                last_speaker = conversation.turns[-1].speaker
                next_speaker = conversation.agent2 if last_speaker == conversation.agent1 else conversation.agent1
            
            other_agent = conversation.agent1 if next_speaker == conversation.agent2 else conversation.agent2
            
            # Generate response
            response = await self._generate_conversation_response(
                next_speaker, other_agent, conversation, memory_streams[next_speaker]
            )
            
            if response:
                conversation.add_turn(next_speaker, response)
                
                # Add to both agents' memories
                await self._add_conversation_memory(
                    next_speaker, f"Said to {other_agent}: '{response}'",
                    memory_streams[next_speaker], conversation.location
                )
                await self._add_conversation_memory(
                    other_agent, f"{next_speaker} said: '{response}'",
                    memory_streams[other_agent], conversation.location
                )
                
                return True
        
        except Exception as e:
            logger.error(f"Error continuing conversation: {e}")
        
        return False
    
    async def _generate_conversation_response(self, speaker: str, other_agent: str,
                                            conversation: Conversation,
                                            memory_stream: MemoryStream) -> str:
        """Generate a conversation response."""
        try:
            # Use conversation engine (handles committee vs single-model)
            # Note: The query fix is already implemented in the conversation engine methods
            conv_engine = ConversationEngine.get_engine()
            return await conv_engine.generate_response(speaker, other_agent, conversation, memory_stream)
            
        except Exception as e:
            logger.error(f"Error generating conversation response: {e}")
            return "I see."
    
    async def should_end_conversation(self, conversation: Conversation) -> bool:
        """Determine if a conversation should end (rule-based, no LLM)."""
        if len(conversation.turns) < 2:
            return False
        return len(conversation.turns) >= cfg.MAX_CONVERSATION_TURNS
    
    async def end_conversation(self, conversation: Conversation,
                              memory_streams: dict[str, MemoryStream],
                              skill_banks: dict[str, SkillBank] | None = None,
                              agents: dict | None = None,
                              current_time=None):
        """End an active conversation, distill skills, and trigger re-planning."""
        conversation.active = False
        
        # Remove from active conversations
        key = self.get_conversation_key(conversation.agent1, conversation.agent2)
        if key in self.active_conversations:
            del self.active_conversations[key]
        
        # Add to history
        self.conversation_history.append(conversation)
        
        # Add conversation summary to both agents' memories
        await self._add_conversation_memory(
            conversation.agent1, f"Finished conversation with {conversation.agent2}",
            memory_streams[conversation.agent1], conversation.location
        )
        await self._add_conversation_memory(
            conversation.agent2, f"Finished conversation with {conversation.agent1}",
            memory_streams[conversation.agent2], conversation.location
        )
        
        logger.info(f"Ended conversation between {conversation.agent1} and {conversation.agent2}")
        
        # Generate voice audio for the conversation (non-blocking)
        if voice_conversation_end is not None:
            try:
                audio_path = await voice_conversation_end(conversation)
                if audio_path:
                    logger.info(f"Voice audio generated: {audio_path}")
            except Exception as e:
                logger.warning(f"Voice generation failed (non-critical): {e}")
        
        # Trigger reactive re-planning for both participants
        if agents and current_time and conversation.turns:
            conv_summary = conversation.get_history_text()
            for agent_name, other_name in [
                (conversation.agent1, conversation.agent2),
                (conversation.agent2, conversation.agent1)
            ]:
                if agent_name in agents:
                    try:
                        changed = await agents[agent_name].react_to_conversation(
                            other_name, conv_summary, current_time
                        )
                        if changed:
                            logger.info(f"[replan] {agent_name} modified plans after talking to {other_name}")
                    except Exception as e:
                        logger.error(f"Error in reactive re-planning for {agent_name}: {e}")
        
        # Skill distillation disabled for performance
        # if skill_banks:
        #     outcome = "success" if len(conversation.turns) >= 3 else "failure"
        #     conv_text = conversation.get_history_text()
        #     for agent_name in [conversation.agent1, conversation.agent2]:
        #         if agent_name in skill_banks:
        #             asyncio.create_task(self._distill_and_store_skill(...))

    async def _distill_and_store_skill(self, skill_bank: SkillBank, agent_name: str,
                                       conv_text: str, outcome: str, location: str):
        """Background: distill a conversation into a skill and store it."""
        skill = await distill_conversation_skill(agent_name, conv_text, outcome, location)
        if skill:
            skill_bank.add_skill(skill)
    
    async def _add_conversation_memory(self, agent_name: str, description: str,
                                     memory_stream: MemoryStream, location: str):
        """Add a conversation memory to an agent's memory stream."""
        memory = Memory(
            agent_name=agent_name,
            description=description,
            memory_type="observation",
            importance_score=6,  # Conversations are moderately important
            location=location
        )
        memory_stream.add_memory(memory)
    
    async def update_conversations(self, memory_streams: dict[str, MemoryStream],
                                   skill_banks: dict[str, SkillBank] | None = None,
                                   agents: dict | None = None,
                                   current_time=None,
                                   current_tick: int = 0):
        """Update all active conversations."""
        self._current_tick = current_tick
        conversations_to_end = []
        
        for conversation in list(self.active_conversations.values()):
            if conversation.active:
                # Continue conversation
                continued = await self.continue_conversation(conversation, memory_streams)
                
                if not continued or await self.should_end_conversation(conversation):
                    conversations_to_end.append(conversation)
        
        # End conversations that should end
        for conversation in conversations_to_end:
            await self.end_conversation(conversation, memory_streams, skill_banks, 
                                       agents=agents, current_time=current_time)
    
    def get_active_conversations_summary(self) -> list[str]:
        """Get summary of all active conversations."""
        summaries = []
        for conversation in self.active_conversations.values():
            location = conversation.location
            participants = f"{conversation.agent1} and {conversation.agent2}"
            turn_count = len(conversation.turns)
            summaries.append(f"{participants} talking at {location} ({turn_count} turns)")
        return summaries
