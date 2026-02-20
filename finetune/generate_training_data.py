#!/usr/bin/env python3
"""Generate fine-tuning training data for Smallville character actor model.

Uses rich YAML profiles + Ollama to generate in-character conversations
across 8 scenario types. Outputs ShareGPT-format JSONL for unsloth.
"""

import yaml
import json
import os
import sys
import random
import asyncio
import aiohttp
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).parent / "profiles"
OUTPUT_DIR = Path(__file__).parent / "data"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Models to try in order
MODELS = [
    "qwen3:14b",           # Best local quality
    "gemma3:4b",            # Fallback
]

# 8 scenario types with templates
SCENARIOS = {
    "casual_greeting": {
        "description": "Running into someone on the street or at a familiar location",
        "prompt_template": (
            "Write a short conversation (3-4 turns) between {agent_a} and {agent_b} "
            "who run into each other at {location}. They exchange greetings and brief small talk. "
            "Each character must sound COMPLETELY DIFFERENT based on their profiles.\n\n"
            "{agent_a_profile}\n\n{agent_b_profile}\n\n"
            "Rules:\n"
            "- Output ONLY the dialogue, alternating speakers\n"
            "- Format: 'Name: dialogue' per line\n"
            "- 3-4 turns each (6-8 lines total)\n"
            "- Each character must use their unique speech patterns, vocabulary, and catchphrases\n"
            "- No narration, no stage directions, no actions in parentheses"
        ),
        "locations": ["Johnson Park", "Hobbs Cafe", "Harvey Oak Supply Store", "the Library", "Oak Hill College", "Town Hall"]
    },
    "work_talk": {
        "description": "Discussing their profession or daily work",
        "prompt_template": (
            "Write a conversation (3-4 turns each) between {agent_a} and {agent_b} "
            "where they discuss {agent_a}'s work. {agent_a} is talking about their job "
            "and {agent_b} is responding in their own characteristic way.\n\n"
            "{agent_a_profile}\n\n{agent_b_profile}\n\n"
            "Rules:\n"
            "- Output ONLY the dialogue, alternating speakers\n"
            "- Format: 'Name: dialogue' per line\n"
            "- 3-4 turns each\n"
            "- {agent_a} should use work-specific vocabulary from their profile\n"
            "- {agent_b} should respond in THEIR unique voice, not mimic {agent_a}\n"
            "- No narration, no stage directions"
        ),
    },
    "event_sharing": {
        "description": "One agent tells another about the Valentine's Day party",
        "prompt_template": (
            "Write a conversation (3-4 turns each) where {agent_a} tells {agent_b} about "
            "Isabella Rodriguez's Valentine's Day party at Hobbs Cafe (Feb 14, 5-7 PM). "
            "{agent_a} heard about it recently and is sharing the news.\n\n"
            "{agent_a_profile}\n\n{agent_b_profile}\n\n"
            "Rules:\n"
            "- Output ONLY the dialogue, alternating speakers\n"
            "- Format: 'Name: dialogue' per line\n"
            "- 3-4 turns each\n"
            "- {agent_a} shares the party info in THEIR unique voice\n"
            "- {agent_b} reacts in THEIR unique voice (excited, cautious, practical, etc.)\n"
            "- The party details should be accurate: Hobbs Cafe, Feb 14, 5-7 PM, Valentine's theme\n"
            "- No narration, no stage directions"
        ),
    },
    "responding_to_invitation": {
        "description": "Being directly invited to something and deciding",
        "prompt_template": (
            "Write a conversation (3-4 turns each) where {agent_a} directly invites {agent_b} "
            "to a community gathering at {location}. {agent_b} considers whether to attend "
            "based on their personality and schedule.\n\n"
            "{agent_a_profile}\n\n{agent_b_profile}\n\n"
            "Rules:\n"
            "- Output ONLY the dialogue, alternating speakers\n"
            "- Format: 'Name: dialogue' per line\n"
            "- 3-4 turns each\n"
            "- {agent_b}'s response should reflect their personality: social people say yes eagerly, "
            "busy people negotiate timing, introverts hesitate but consider it\n"
            "- No narration, no stage directions"
        ),
        "locations": ["Hobbs Cafe", "Johnson Park", "The Rose and Crown Pub", "the Library", "Town Hall"]
    },
    "emotional_moment": {
        "description": "Sharing good or bad news, showing emotional depth",
        "prompt_template": (
            "Write a conversation (3-4 turns each) where {agent_a} shares {emotion_context} "
            "with {agent_b}. This is a genuine emotional moment between them.\n\n"
            "{agent_a_profile}\n\n{agent_b_profile}\n\n"
            "Rules:\n"
            "- Output ONLY the dialogue, alternating speakers\n"
            "- Format: 'Name: dialogue' per line\n"
            "- 3-4 turns each\n"
            "- {agent_a} should express emotion in THEIR characteristic way (per their emotional range profile)\n"
            "- {agent_b} should respond supportively in THEIR own voice\n"
            "- Keep it authentic — not melodramatic\n"
            "- No narration, no stage directions"
        ),
        "emotions": [
            "exciting news about a personal achievement",
            "worry about a family member",
            "frustration about something at work",
            "gratitude for the other person's support",
            "nostalgia about the past",
            "anxiety about the future",
        ]
    },
    "multi_turn_sustained": {
        "description": "Longer conversation maintaining character over 5-6 turns",
        "prompt_template": (
            "Write a longer conversation (5-6 turns each) between {agent_a} and {agent_b} "
            "at {location}. They discuss {topic}. The conversation should flow naturally "
            "and both characters must maintain their distinctive voices throughout.\n\n"
            "{agent_a_profile}\n\n{agent_b_profile}\n\n"
            "Rules:\n"
            "- Output ONLY the dialogue, alternating speakers\n"
            "- Format: 'Name: dialogue' per line\n"
            "- 5-6 turns EACH (10-12 lines total)\n"
            "- Characters must stay in-character throughout — no voice drift\n"
            "- Let the conversation evolve naturally with topic shifts\n"
            "- No narration, no stage directions"
        ),
        "topics": [
            "what Smallville means to them",
            "their hopes for the future",
            "a funny thing that happened recently",
            "changes they'd like to see in town",
            "a shared memory or experience",
        ],
        "locations": ["Hobbs Cafe", "Johnson Park", "The Rose and Crown Pub", "the Library"]
    },
    "disagreement": {
        "description": "Politely disagreeing or pushing back on something",
        "prompt_template": (
            "Write a conversation (3-4 turns each) where {agent_a} and {agent_b} "
            "have a mild disagreement about {disagreement_topic}. Each handles conflict "
            "in their characteristic way — some are direct, some redirect, some use humor.\n\n"
            "{agent_a_profile}\n\n{agent_b_profile}\n\n"
            "Rules:\n"
            "- Output ONLY the dialogue, alternating speakers\n"
            "- Format: 'Name: dialogue' per line\n"
            "- 3-4 turns each\n"
            "- This is a MILD disagreement, not a fight — they remain respectful\n"
            "- Each character disagrees in THEIR way (per disagree_style in profile)\n"
            "- No narration, no stage directions"
        ),
        "topics": [
            "whether the town needs more modern development",
            "the best approach to a community project",
            "how to handle a local issue",
            "priorities for town spending",
            "whether tradition or change is more important",
        ]
    },
    "gossip_relay": {
        "description": "Relaying secondhand information — testing information fidelity",
        "prompt_template": (
            "Write a conversation (3-4 turns each) where {agent_a} tells {agent_b} "
            "about something they heard from someone else: {gossip_content}. "
            "Each character filters and shares information in their characteristic way.\n\n"
            "{agent_a_profile}\n\n{agent_b_profile}\n\n"
            "Rules:\n"
            "- Output ONLY the dialogue, alternating speakers\n"
            "- Format: 'Name: dialogue' per line\n"
            "- 3-4 turns each\n"
            "- {agent_a} relays the info in THEIR voice — some embellish, some are factual, some are cautious\n"
            "- {agent_b} reacts in THEIR characteristic way\n"
            "- No narration, no stage directions"
        ),
        "gossip": [
            "Isabella is planning an elaborate Valentine's Day party with wildflower decorations",
            "Sam Moore is planning a big campaign event next week",
            "Dr. Williams has been checking on Frank Wilson more than usual",
            "Carlos and Eddy are planning a combined art-music performance",
            "The old mill area might be getting redeveloped",
            "Rachel Kim designed new logos for several local businesses",
        ]
    },
}

# All agent names for pairing
ALL_AGENTS = [
    "John Lin", "Mei Lin", "Eddy Lin", "Isabella Rodriguez", "Tom Moreno",
    "Sam Moore", "Carmen Moreno", "Carlos Gomez", "Maria Santos", "Sarah Chen",
    "Mike Johnson", "Jennifer Moore", "Emily Moore", "Diego Moreno", "Ana Santos",
    "Dr. Williams", "Professor Anderson", "Professor Davis", "Lisa Park",
    "Mayor Johnson", "Miguel Rodriguez", "Mrs. Peterson", "Officer Thompson",
    "Rachel Kim", "Frank Wilson"
]


def load_profile(agent_name: str) -> Dict[str, Any]:
    """Load a YAML profile for an agent."""
    filename = agent_name.lower().replace(" ", "_").replace(".", "") + ".yaml"
    filepath = PROFILES_DIR / filename
    if not filepath.exists():
        logger.warning(f"Profile not found: {filepath}")
        return {}
    with open(filepath) as f:
        return yaml.safe_load(f)


def format_profile_for_prompt(profile: Dict[str, Any]) -> str:
    """Format a profile into a concise prompt section."""
    if not profile:
        return "No profile available."
    
    sections = []
    sections.append(f"=== {profile.get('name', 'Unknown')} ===")
    sections.append(f"Age: {profile.get('age', '?')} | Occupation: {profile.get('occupation', '?')}")
    
    if 'speech_style' in profile:
        sections.append(f"Speech style: {profile['speech_style'].strip()}")
    if 'vocabulary_level' in profile:
        sections.append(f"Vocabulary: {profile['vocabulary_level']}")
    if 'catchphrases' in profile:
        sections.append(f"Catchphrases: {', '.join(profile['catchphrases'][:3])}")
    if 'sentence_length' in profile:
        sections.append(f"Sentence length: {profile['sentence_length']}")
    if 'humor_style' in profile:
        sections.append(f"Humor: {profile['humor_style']}")
    if 'default_mood' in profile:
        sections.append(f"Default mood: {profile['default_mood']}")
    if 'greeting_style' in profile:
        sections.append(f"Greeting: {profile['greeting_style']}")
    if 'disagree_style' in profile:
        sections.append(f"Disagree style: {profile['disagree_style']}")
    if 'information_sharing' in profile:
        sections.append(f"Info sharing: {profile['information_sharing']}")
    if 'quirks' in profile:
        sections.append(f"Quirks: {', '.join(profile['quirks'][:3])}")
    if 'example_lines' in profile:
        sections.append(f"Example lines:\n" + "\n".join(f'  - "{l}"' for l in profile['example_lines'][:2]))
    
    return "\n".join(sections)


def build_system_prompt(profile: Dict[str, Any], mood: str = None, talking_to: str = None) -> str:
    """Build the structured system prompt that will be used during both training and inference."""
    if not profile:
        return ""
    
    parts = []
    parts.append(f"[AGENT: {profile.get('name', 'Unknown')}]")
    parts.append(f"[AGE: {profile.get('age', '?')}] [OCCUPATION: {profile.get('occupation', '?')}]")
    
    if 'speech_style' in profile:
        style = profile['speech_style'].strip().replace('\n', ' ')
        parts.append(f"[SPEECH: {style}]")
    if 'vocabulary_level' in profile:
        parts.append(f"[VOCABULARY: {profile['vocabulary_level']}]")
    if 'catchphrases' in profile:
        parts.append(f"[CATCHPHRASES: {' | '.join(profile['catchphrases'])}]")
    if 'sentence_length' in profile:
        parts.append(f"[SENTENCE_LENGTH: {profile['sentence_length']}]")
    
    mood_text = mood or profile.get('default_mood', '')
    if mood_text:
        parts.append(f"[MOOD: {mood_text}]")
    if talking_to:
        parts.append(f"[TALKING_TO: {talking_to}]")
    if 'humor_style' in profile:
        parts.append(f"[HUMOR: {profile['humor_style']}]")
    if 'quirks' in profile:
        parts.append(f"[QUIRKS: {' | '.join(profile['quirks'][:3])}]")
    
    return "\n".join(parts)


def parse_dialogue(raw_text: str, agent_a: str, agent_b: str) -> List[Dict[str, str]]:
    """Parse raw dialogue text into structured turns."""
    turns = []
    current_speaker = None
    current_text = []
    
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts with a speaker name
        speaker_found = None
        for name in [agent_a, agent_b]:
            if line.startswith(f"{name}:"):
                speaker_found = name
                line = line[len(name)+1:].strip()
                break
            # Also check first name only
            first_name = name.split()[0]
            if line.startswith(f"{first_name}:"):
                speaker_found = name
                line = line[len(first_name)+1:].strip()
                break
        
        if speaker_found:
            # Save previous turn
            if current_speaker and current_text:
                turns.append({"speaker": current_speaker, "text": " ".join(current_text)})
            current_speaker = speaker_found
            current_text = [line] if line else []
        elif current_speaker:
            current_text.append(line)
    
    # Save last turn
    if current_speaker and current_text:
        turns.append({"speaker": current_speaker, "text": " ".join(current_text)})
    
    return turns


def dialogue_to_training_examples(turns: List[Dict], agent_a: str, agent_b: str,
                                    profile_a: Dict, profile_b: Dict,
                                    mood_a: str = None, mood_b: str = None) -> List[Dict]:
    """Convert parsed dialogue turns into ShareGPT training examples.
    
    Each agent's lines become separate training examples with their own system prompt.
    """
    examples = []
    
    for i, turn in enumerate(turns):
        speaker = turn["speaker"]
        text = turn["text"].strip().strip('"').strip("'")
        
        if not text:
            continue
        
        # Determine which profile/mood to use
        if speaker == agent_a:
            profile = profile_a
            mood = mood_a
            other = agent_b
        elif speaker == agent_b:
            profile = profile_b
            mood = mood_b
            other = agent_a
        else:
            continue
        
        # Build context from previous turns
        context_lines = []
        for prev_turn in turns[:i]:
            context_lines.append(f"{prev_turn['speaker']}: {prev_turn['text']}")
        
        situation = "\n".join(context_lines[-4:]) if context_lines else "Starting a new conversation."
        
        system_prompt = build_system_prompt(profile, mood=mood, talking_to=other)
        
        example = {
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": situation if context_lines else f"{other} approaches you for a conversation."},
                {"role": "assistant", "content": text}
            ]
        }
        examples.append(example)
    
    return examples


async def generate_dialogue(session: aiohttp.ClientSession, prompt: str, model: str) -> Optional[str]:
    """Call Ollama to generate dialogue."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.85,
            "num_predict": 500,
        }
    }
    
    try:
        async with session.post(OLLAMA_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("message", {}).get("content", "").strip()
            elif resp.status == 429:
                logger.warning(f"Rate limited on {model}, waiting...")
                await asyncio.sleep(5)
                return None
            else:
                logger.warning(f"Ollama returned {resp.status}")
                return None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


async def generate_scenario(session: aiohttp.ClientSession, scenario_type: str, 
                             agent_a: str, agent_b: str, model: str,
                             extra_vars: Dict = None) -> List[Dict]:
    """Generate a single scenario conversation and return training examples."""
    scenario = SCENARIOS[scenario_type]
    profile_a = load_profile(agent_a)
    profile_b = load_profile(agent_b)
    
    if not profile_a or not profile_b:
        return []
    
    # Build the prompt
    template_vars = {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "agent_a_profile": format_profile_for_prompt(profile_a),
        "agent_b_profile": format_profile_for_prompt(profile_b),
    }
    
    # Add scenario-specific variables
    if "locations" in scenario:
        template_vars["location"] = random.choice(scenario["locations"])
    if "topics" in scenario:
        template_vars["topic"] = random.choice(scenario["topics"])
    if "emotions" in scenario:
        template_vars["emotion_context"] = random.choice(scenario["emotions"])
    if "topics" in scenario and scenario_type == "disagreement":
        template_vars["disagreement_topic"] = random.choice(scenario["topics"])
    if "gossip" in scenario:
        template_vars["gossip_content"] = random.choice(scenario["gossip"])
    
    if extra_vars:
        template_vars.update(extra_vars)
    
    prompt = scenario["prompt_template"].format(**template_vars)
    
    # Generate
    raw = await generate_dialogue(session, prompt, model)
    if not raw:
        return []
    
    # Parse dialogue
    turns = parse_dialogue(raw, agent_a, agent_b)
    if len(turns) < 4:
        logger.warning(f"Too few turns ({len(turns)}) for {scenario_type}: {agent_a} + {agent_b}")
        return []
    
    # Convert to training examples
    examples = dialogue_to_training_examples(turns, agent_a, agent_b, profile_a, profile_b)
    return examples


def generate_agent_pairs(n_per_scenario: int = 8) -> List[tuple]:
    """Generate diverse agent pairs ensuring each agent appears multiple times."""
    pairs = []
    
    # Ensure relationship pairs are included
    relationship_pairs = [
        ("John Lin", "Mei Lin"),          # married
        ("Eddy Lin", "Maria Santos"),     # dating
        ("Eddy Lin", "Carlos Gomez"),     # best friends
        ("Isabella Rodriguez", "Miguel Rodriguez"),  # siblings
        ("Isabella Rodriguez", "Sarah Chen"),        # best friends
        ("Tom Moreno", "Carmen Moreno"),   # married
        ("Sam Moore", "Jennifer Moore"),  # married
        ("Sam Moore", "Mayor Johnson"),   # rivals
        ("Sarah Chen", "Mike Johnson"),   # dating
        ("Diego Moreno", "Emily Moore"),  # friends
        ("Maria Santos", "Ana Santos"),   # sisters
        ("Maria Santos", "Dr. Williams"), # mentor
        ("Carlos Gomez", "Professor Davis"), # mentor
        ("Isabella Rodriguez", "Rachel Kim"), # friends
        ("Mrs. Peterson", "Carmen Moreno"),   # teacher bond
    ]
    pairs.extend(relationship_pairs)
    
    # Add random pairs to diversify
    used = set(tuple(sorted(p)) for p in pairs)
    agents = ALL_AGENTS.copy()
    
    while len(pairs) < n_per_scenario * len(SCENARIOS):
        a, b = random.sample(agents, 2)
        key = tuple(sorted([a, b]))
        if key not in used:
            pairs.append((a, b))
            used.add(key)
    
    random.shuffle(pairs)
    return pairs


async def main():
    parser = argparse.ArgumentParser(description="Generate training data for Smallville character actor")
    parser.add_argument("--model", default="qwen3:14b", help="Ollama model to use")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "training_data.jsonl"), help="Output file")
    parser.add_argument("--conversations", type=int, default=200, help="Target number of conversations")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent Ollama requests")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without generating")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate plan
    pairs = generate_agent_pairs(n_per_scenario=args.conversations // len(SCENARIOS))
    scenario_types = list(SCENARIOS.keys())
    
    # Build work items: (scenario_type, agent_a, agent_b)
    work_items = []
    pair_idx = 0
    for i in range(args.conversations):
        scenario = scenario_types[i % len(scenario_types)]
        a, b = pairs[pair_idx % len(pairs)]
        work_items.append((scenario, a, b))
        pair_idx += 1
    
    logger.info(f"Plan: {len(work_items)} conversations across {len(SCENARIOS)} scenario types")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")
    
    if args.dry_run:
        for scenario, a, b in work_items[:20]:
            print(f"  {scenario}: {a} + {b}")
        print(f"  ... and {len(work_items) - 20} more")
        return
    
    # Generate
    all_examples = []
    semaphore = asyncio.Semaphore(args.max_concurrent)
    completed = 0
    failed = 0
    
    async with aiohttp.ClientSession() as session:
        async def process_item(item):
            nonlocal completed, failed
            scenario, a, b = item
            async with semaphore:
                examples = await generate_scenario(session, scenario, a, b, args.model)
                if examples:
                    all_examples.extend(examples)
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{len(work_items)} conversations, {len(all_examples)} training examples")
                else:
                    failed += 1
        
        # Process in batches to avoid overwhelming Ollama
        batch_size = 4
        for i in range(0, len(work_items), batch_size):
            batch = work_items[i:i+batch_size]
            tasks = [process_item(item) for item in batch]
            await asyncio.gather(*tasks)
    
    # Save
    with open(args.output, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")
    
    logger.info(f"Done! {completed} conversations → {len(all_examples)} training examples")
    logger.info(f"Failed: {failed}")
    logger.info(f"Saved to: {args.output}")
    
    # Stats
    agent_counts = {}
    for ex in all_examples:
        system = ex["conversations"][0]["content"]
        for agent in ALL_AGENTS:
            if agent in system:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
                break
    
    logger.info("Examples per agent:")
    for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {agent}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
