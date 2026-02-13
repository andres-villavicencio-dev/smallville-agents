"""
Committee of Experts — Mixture of small specialized models for agent cognition.

Sequential execution to respect 8GB VRAM constraints. Each expert runs one at a time,
Ollama handles model swapping. Small models (1B-3B) swap in ~1-2s, larger (4B-7B) ~3-5s.

Architecture:
    Each decision flows through relevant experts sequentially, building context.
    A judge model synthesizes expert outputs into a final action/response.
"""
import logging
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from config import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class ExpertConfig:
    """Configuration for a domain expert."""
    name: str
    model: str
    role: str           # System prompt describing this expert's domain
    max_tokens: int = 150
    temperature: float = 0.7


# ── Expert Definitions ──────────────────────────────────────────────────────

EXPERTS = {
    "social": ExpertConfig(
        name="Social",
        model=os.getenv("COMMITTEE_MODEL_SOCIAL", "llama3.2:3b"),
        role=(
            "You are a social reasoning expert. You analyze relationships, social norms, "
            "and interpersonal dynamics. Given a situation, assess: Who should this person "
            "interact with? What social obligations exist? What would be socially appropriate? "
            "Be concise — 1-2 sentences max."
        ),
        temperature=0.7,
    ),
    "spatial": ExpertConfig(
        name="Spatial",
        model=os.getenv("COMMITTEE_MODEL_SPATIAL", "qwen2.5:3b"),
        role=(
            "You are a spatial reasoning expert. You analyze locations, distances, and movement. "
            "Given a situation, assess: Where is this person? What locations are nearby? "
            "Where should they go next based on their goals? "
            "Be concise — 1-2 sentences max."
        ),
        temperature=0.5,
    ),
    "temporal": ExpertConfig(
        name="Temporal",
        model=os.getenv("COMMITTEE_MODEL_TEMPORAL", "gemma3:1b"),
        role=(
            "You are a time and scheduling expert. You analyze time-of-day, deadlines, "
            "and temporal priorities. Given a situation, assess: What time is it? "
            "What's urgent? What should happen next chronologically? "
            "Be concise — 1 sentence max."
        ),
        max_tokens=60,
        temperature=0.3,
    ),
    "emotional": ExpertConfig(
        name="Emotional",
        model=os.getenv("COMMITTEE_MODEL_EMOTIONAL", "llama3.2:3b"),
        role=(
            "You are an emotional intelligence expert. You analyze mood, personality, "
            "and emotional states. Given a situation, assess: How is this person feeling? "
            "What emotions drive their next action? How does their personality shape their response? "
            "Be concise — 1-2 sentences max."
        ),
        temperature=0.8,
    ),
    "memory": ExpertConfig(
        name="Memory",
        model=os.getenv("COMMITTEE_MODEL_MEMORY", "gemma3:4b"),
        role=(
            "You are a memory and context expert. You analyze past experiences and their relevance. "
            "Given memories and a current situation, assess: What past experience is most relevant? "
            "What has this person committed to? What patterns exist in their behavior? "
            "Be concise — 1-2 sentences max."
        ),
        temperature=0.6,
    ),
    "dialogue": ExpertConfig(
        name="Dialogue",
        model=os.getenv("COMMITTEE_MODEL_DIALOGUE", "llama3.2:3b"),
        role=(
            "You are a dialogue expert. You generate natural, character-consistent conversation. "
            "Given context about a person and situation, generate what they would actually say. "
            "Keep it natural and brief — 1-2 sentences of dialogue only."
        ),
        temperature=0.9,
        max_tokens=100,
    ),
    "judge": ExpertConfig(
        name="Judge",
        model=os.getenv("COMMITTEE_MODEL_JUDGE", "qwen2.5:3b"),
        role=(
            "You are a synthesis expert. You receive assessments from multiple domain experts "
            "and combine them into a single coherent action or response. "
            "Resolve any conflicts between experts by weighing relevance to the situation. "
            "Output a clear, specific action or response — 1-2 sentences max."
        ),
        temperature=0.5,
        max_tokens=150,
    ),
}


# ── Expert Pipelines ────────────────────────────────────────────────────────
# Define which experts run (in order) for each cognitive task.

PIPELINES = {
    "decide_action": ["temporal", "spatial", "social", "memory", "emotional", "judge"],
    "conversation_response": ["memory", "emotional", "dialogue"],
    "should_converse": ["social", "emotional", "judge"],
    "plan_day": ["temporal", "memory", "spatial", "judge"],
    "reflect": ["memory", "emotional", "judge"],
}


import requests

# Import status callback
try:
    from llm import _notify_llm_status
except ImportError:
    def _notify_llm_status(a, t, m): pass


class Committee:
    """Orchestrates sequential expert consultations for agent decisions."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self._call_count = 0
        self._expert_stats: Dict[str, int] = {}
        self._shutdown_flag = False

    def shutdown(self):
        """Signal the committee to stop mid-pipeline."""
        self._shutdown_flag = True

    async def consult(
        self,
        pipeline_name: str,
        context: str,
        agent_name: str = "",
        extra: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Run a named pipeline of experts sequentially.

        Args:
            pipeline_name: Key into PIPELINES dict
            context: The situation description all experts receive
            agent_name: Name of the agent making the decision
            extra: Additional context per-expert (e.g. {"memory": "relevant memories here"})

        Returns:
            Final synthesized output from the judge/last expert.
        """
        pipeline = PIPELINES.get(pipeline_name)
        if not pipeline:
            logger.error(f"Unknown pipeline: {pipeline_name}")
            return ""

        expert_outputs: Dict[str, str] = {}
        extra = extra or {}

        for expert_key in pipeline:
            expert = EXPERTS.get(expert_key)
            if not expert:
                logger.warning(f"Unknown expert: {expert_key}")
                continue

            # Build the prompt for this expert
            prompt = self._build_expert_prompt(
                expert_key, expert, context, agent_name, expert_outputs, extra
            )

            # Check if simulation is still running
            if self._shutdown_flag:
                break

            # Notify display & call the model
            _notify_llm_status(agent_name, f"{expert.name} expert", expert.model)
            output = await self._call_model(expert, prompt)
            if output:
                expert_outputs[expert_key] = output
                logger.debug(f"[{agent_name}] {expert.name} expert: {output[:80]}...")

            self._expert_stats[expert_key] = self._expert_stats.get(expert_key, 0) + 1
            self._call_count += 1

        # Return the last expert's output (usually the judge)
        last_expert = pipeline[-1]
        return expert_outputs.get(last_expert, "")

    def _build_expert_prompt(
        self,
        expert_key: str,
        expert: ExpertConfig,
        context: str,
        agent_name: str,
        previous_outputs: Dict[str, str],
        extra: Dict[str, str],
    ) -> str:
        """Build the prompt for an expert, including previous expert outputs."""
        parts = []

        if agent_name:
            parts.append(f"Agent: {agent_name}")

        parts.append(f"Situation: {context}")

        # Add any extra context specific to this expert
        if expert_key in extra:
            parts.append(f"Additional context: {extra[expert_key]}")

        # For the judge (or any expert after the first), include previous assessments
        if previous_outputs:
            parts.append("\nPrevious expert assessments:")
            for prev_key, prev_output in previous_outputs.items():
                prev_expert = EXPERTS.get(prev_key)
                if prev_expert:
                    parts.append(f"  [{prev_expert.name}]: {prev_output}")

        parts.append("\nYour assessment:")

        return "\n".join(parts)

    async def _call_model(self, expert: ExpertConfig, prompt: str) -> str:
        """Call an Ollama model. Sequential — one at a time for VRAM."""
        try:
            payload = {
                "model": expert.model,
                "messages": [
                    {"role": "system", "content": expert.role},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": expert.temperature,
                    "num_predict": expert.max_tokens,
                },
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            return result["message"]["content"].strip()

        except Exception as e:
            logger.error(f"{expert.name} expert ({expert.model}) failed: {e}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get committee usage statistics."""
        return {
            "total_calls": self._call_count,
            "per_expert": dict(self._expert_stats),
        }

    def print_stats(self):
        """Print committee stats to log."""
        stats = self.get_stats()
        logger.info(f"Committee stats: {stats['total_calls']} total calls")
        for expert, count in stats["per_expert"].items():
            model = EXPERTS[expert].model
            logger.info(f"  {expert}: {count} calls ({model})")


# ── Convenience Functions ───────────────────────────────────────────────────

# Global committee instance
_committee: Optional[Committee] = None


def get_committee() -> Committee:
    """Get or create the global committee instance."""
    global _committee
    if _committee is None:
        _committee = Committee()
    return _committee


async def decide_action(agent_name: str, situation: str, memories: str = "") -> str:
    """Decide what an agent should do next using the full expert pipeline."""
    committee = get_committee()
    extra = {}
    if memories:
        extra["memory"] = memories
    return await committee.consult("decide_action", situation, agent_name, extra)


async def generate_dialogue(agent_name: str, situation: str, memories: str = "") -> str:
    """Generate a conversation response using the dialogue pipeline."""
    committee = get_committee()
    extra = {}
    if memories:
        extra["memory"] = memories
    return await committee.consult("conversation_response", situation, agent_name, extra)


async def should_converse(agent_name: str, situation: str) -> bool:
    """Decide whether an agent should initiate conversation."""
    committee = get_committee()
    result = await committee.consult("should_converse", situation, agent_name)
    return "yes" in result.lower()


async def plan_day(agent_name: str, situation: str, memories: str = "") -> str:
    """Generate a daily plan using the planning pipeline."""
    committee = get_committee()
    extra = {}
    if memories:
        extra["memory"] = memories
    return await committee.consult("plan_day", situation, agent_name, extra)


async def reflect(agent_name: str, situation: str, memories: str = "") -> str:
    """Generate a reflection using the reflection pipeline."""
    committee = get_committee()
    extra = {}
    if memories:
        extra["memory"] = memories
    return await committee.consult("reflect", situation, agent_name, extra)


def print_committee_config():
    """Print the committee configuration."""
    print("\n🧠 Committee of Experts (Sequential Mode — 8GB VRAM):")
    print("─" * 55)
    for key, expert in EXPERTS.items():
        print(f"  {expert.name:<12} │ {expert.model:<15} │ temp={expert.temperature}")
    print("─" * 55)
    print("\n📋 Pipelines:")
    for name, experts in PIPELINES.items():
        chain = " → ".join(EXPERTS[e].name for e in experts if e in EXPERTS)
        print(f"  {name}: {chain}")
    print()
