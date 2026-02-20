"""
Committee of Experts — Mixture of small specialized models for agent cognition.

Sequential execution to respect 8GB VRAM constraints. Each expert runs one at a time,
Ollama handles model swapping. Small models (1B-3B) swap in ~1-2s, larger (4B-7B) ~3-5s.

Architecture:
    Each decision flows through relevant experts sequentially, building context.
    A judge model synthesizes expert outputs into a final action/response.
"""
import asyncio
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
        model=os.getenv("COMMITTEE_MODEL_SOCIAL", "smallville-social"),
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
            "You are a dialogue writer. Output ONLY the character's spoken words — nothing else. "
            "No narration, no stage directions, no analysis, no personality descriptions, no predictions. "
            "Do not start with the character's name. Do not explain why they would say it. "
            "Just write 1-2 sentences of natural dialogue as if you ARE the character speaking."
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
    "should_converse": ["social"],
    "plan_day": ["temporal", "memory", "spatial", "judge"],
    "reflect": ["memory", "emotional", "judge"],
}

# Per-pipeline token overrides for the final expert (e.g. judge needs more room for plans)
PIPELINE_TOKEN_OVERRIDES = {
    "plan_day": {"judge": 500},
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

            # Apply per-pipeline token overrides (e.g. judge needs more tokens for plans)
            token_overrides = PIPELINE_TOKEN_OVERRIDES.get(pipeline_name, {})
            effective_max_tokens = token_overrides.get(expert_key, expert.max_tokens)

            # Notify display & call the model
            _notify_llm_status(agent_name, f"{expert.name} expert", expert.model)
            output = await self._call_model(expert, prompt, max_tokens_override=effective_max_tokens)
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

    async def _call_model(self, expert: ExpertConfig, prompt: str, max_tokens_override: int = None) -> str:
        """Call an Ollama model. Sequential — one at a time for VRAM.
        Retries with backoff if Ollama is temporarily unavailable (e.g. during TTS)."""
        import time
        
        effective_max_tokens = max_tokens_override if max_tokens_override is not None else expert.max_tokens
        payload = {
            "model": expert.model,
            "messages": [
                {"role": "system", "content": expert.role},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": expert.temperature,
                "num_predict": effective_max_tokens,
            },
        }

        max_retries = 10
        base_delay = 15  # seconds — Ollama restart takes ~5-10s, TTS pipeline ~5 min
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"].strip()

            except (requests.ConnectionError, requests.Timeout) as e:
                delay = min(base_delay * (attempt + 1), 120)
                logger.warning(
                    f"{expert.name} expert: Ollama unavailable (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay}s... ({type(e).__name__})"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"{expert.name} expert ({expert.model}) failed: {e}")
                return ""
        
        logger.error(f"{expert.name} expert: Ollama unavailable after {max_retries} retries, giving up")
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


def _load_character_profiles() -> Dict[str, Dict]:
    """Load all YAML character profiles once and cache them."""
    profiles_dir = os.path.join(os.path.dirname(__file__), "finetune", "profiles")
    profiles = {}
    if not os.path.isdir(profiles_dir):
        logger.warning(f"Character profiles directory not found: {profiles_dir}")
        return profiles
    import yaml
    for fname in os.listdir(profiles_dir):
        if not fname.endswith(".yaml"):
            continue
        try:
            with open(os.path.join(profiles_dir, fname)) as f:
                p = yaml.safe_load(f)
            if p and "name" in p:
                profiles[p["name"]] = p
        except Exception as e:
            logger.warning(f"Failed to load profile {fname}: {e}")
    logger.info(f"Loaded {len(profiles)} character profiles for dialogue")
    return profiles


_CHARACTER_PROFILES: Optional[Dict[str, Dict]] = None


def _get_character_profiles() -> Dict[str, Dict]:
    global _CHARACTER_PROFILES
    if _CHARACTER_PROFILES is None:
        _CHARACTER_PROFILES = _load_character_profiles()
    return _CHARACTER_PROFILES


def _build_character_system_prompt(agent_name: str, talking_to: str = "someone", mood: str = "neutral") -> Optional[str]:
    """Build the [AGENT:] bracket system prompt the fine-tuned model was trained on."""
    profiles = _get_character_profiles()
    p = profiles.get(agent_name)
    if not p:
        return None
    
    quirks = p.get("quirks", [])
    quirks_str = " | ".join(quirks[:3]) if isinstance(quirks, list) else str(quirks)
    catchphrases = p.get("catchphrases", [])
    catch_str = " | ".join(catchphrases[:3]) if isinstance(catchphrases, list) else str(catchphrases)
    
    return (
        f"[AGENT: {p['name']}]\n"
        f"[AGE: {p.get('age', '?')}] [OCCUPATION: {p.get('occupation', '?')}]\n"
        f"[SPEECH: {p.get('speech_style', 'natural')}]\n"
        f"[VOCABULARY: {p.get('vocabulary_level', 'average')}]\n"
        f"[CATCHPHRASES: {catch_str}]\n"
        f"[SENTENCE_LENGTH: {p.get('sentence_length', 'medium')}]\n"
        f"[MOOD: {mood}]\n"
        f"[TALKING_TO: {talking_to}]\n"
        f"[HUMOR: {p.get('humor_style', 'situational')}]\n"
        f"[QUIRKS: {quirks_str}]"
    )


async def generate_dialogue(agent_name: str, situation: str, memories: str = "",
                            talking_to: str = "someone", mood: str = "neutral") -> str:
    """Generate a conversation response using the fine-tuned character actor model.
    
    Primary: smallville-actor (fine-tuned gemma-2-2b, character-specific voices)
    Fallback: cloud model (qwen3-coder:480b-cloud) for generic dialogue
    """
    actor_model = os.getenv("DIALOGUE_ACTOR_MODEL", "smallville-actor")
    cloud_model = os.getenv("DIALOGUE_CLOUD_MODEL", "qwen3-coder:480b-cloud")
    
    memory_context = f"\nRelevant memories:\n{memories}" if memories else ""
    
    # Try fine-tuned character actor first
    system_prompt = _build_character_system_prompt(agent_name, talking_to, mood)
    
    if system_prompt:
        try:
            user_prompt = f"{situation}{memory_context}"
            
            payload = {
                "model": actor_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_predict": 150,
                },
            }
            
            _notify_llm_status(agent_name, "dialogue (actor)", actor_model)
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            raw = response.json().get("message", {}).get("content", "").strip()
            if raw and len(raw) > 5:
                return _clean_dialogue(raw, agent_name)
            logger.warning(f"Actor model returned empty/short response for {agent_name}, falling back")
        except Exception as e:
            logger.warning(f"Actor model failed for {agent_name}: {e}, falling back to cloud")
    
    # Fallback: cloud model with generic prompt
    prompt = (
        f"You are {agent_name} in a small town called Smallville. "
        f"You must respond IN CHARACTER as {agent_name}.\n\n"
        f"Current situation:\n{situation}{memory_context}\n\n"
        f"Generate {agent_name}'s next line of dialogue. Rules:\n"
        f"- Output ONLY the spoken words, nothing else\n"
        f"- 1-2 sentences, natural and in-character\n"
        f"- No narration, no stage directions, no analysis\n"
        f"- Do not start with the character's name\n"
        f"- Consider their emotional state and past experiences"
    )
    
    try:
        payload = {
            "model": cloud_model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.9,
                "num_predict": 100,
            },
        }
        
        _notify_llm_status(agent_name, "dialogue (cloud)", cloud_model)
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("message", {}).get("content", "").strip()
        return _clean_dialogue(raw, agent_name)
    except Exception as e:
        logger.warning(f"Cloud dialogue also failed for {agent_name}: {e}, using local fallback")
        committee = get_committee()
        extra = {}
        if memories:
            extra["memory"] = memories
        raw = await committee.consult("conversation_response", situation, agent_name, extra)
        return _clean_dialogue(raw, agent_name)


def _clean_dialogue(text: str, agent_name: str) -> str:
    """Strip meta-commentary leaked by the dialogue expert, keeping only spoken words."""
    if not text:
        return "I see."
    import re
    # Remove common meta prefixes the model leaks
    meta_patterns = [
        rf"^{re.escape(agent_name)}(?:'s)?\s*(?:next line of dialogue|response|would (?:say|respond))[^\"]*?[,:\n]+\s*",
        r"^(?:Based on|Given|Considering|Here is|The response|I predict)[^\"]*?[:\n]+\s*",
        r"^(?:This (?:response|line|dialogue))\s.*$",
    ]
    cleaned = text.strip()
    for pattern in meta_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    # Strip trailing meta-commentary after the actual dialogue
    # Look for sentences starting with "This response/line/aligns/reflects"
    cleaned = re.split(r"\n\s*(?:This (?:response|line|dialogue)|This aligns|This reflects)", cleaned, flags=re.IGNORECASE)[0].strip()
    # Remove wrapping quotes if present
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned if cleaned else "I see."


async def should_converse(agent_name: str, situation: str) -> bool:
    """Decide whether an agent should initiate conversation."""
    committee = get_committee()
    # Use a single fast Social expert call with explicit YES/NO instruction
    expert = EXPERTS["social"]
    prompt = (
        f"Agent: {agent_name}\n"
        f"Situation: {situation}\n\n"
        f"Should {agent_name} initiate a conversation with the other person? "
        f"Consider: Do they have a reason to talk? Are they at the same location? "
        f"Is it socially natural for people in this situation to chat?\n\n"
        f"Respond with YES or NO on the first line, then a brief reason."
    )
    try:
        from llm import _notify_llm_status
    except ImportError:
        _notify_llm_status = lambda a, t, m: None
    _notify_llm_status(agent_name, "should_converse", expert.model)
    result = await committee._call_model(expert, prompt)
    should = result.strip().upper().startswith("YES") if result else False
    logger.info(f"[should_converse] {agent_name}: {'YES' if should else 'NO'} — {result[:100] if result else 'empty'}")
    return should


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
