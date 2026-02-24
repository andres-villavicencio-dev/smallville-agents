"""Voice mapping for Smallville agents → KittenTTS Nano voices.

8 voices available (4M, 4F), 25 agents. We cycle through voices
within each gender, giving different-pitched voices to agents who
frequently interact to maximize distinguishability.

KittenTTS endpoint: http://192.168.1.70:8377/tts
"""

import hashlib
import asyncio
import aiohttp
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

KITTENTTS_URL = os.getenv("KITTENTTS_URL", "http://192.168.1.70:8377/tts")

MALE_VOICES = ["expr-voice-2-m", "expr-voice-3-m", "expr-voice-4-m", "expr-voice-5-m"]
FEMALE_VOICES = ["expr-voice-2-f", "expr-voice-3-f", "expr-voice-4-f", "expr-voice-5-f"]

# Gender mapping for all 25 Smallville agents
AGENT_GENDER = {
    "John Lin": "M",
    "Mei Lin": "F",
    "Eddy Lin": "M",
    "Isabella Rodriguez": "F",
    "Tom Moreno": "M",
    "Sam Moore": "M",
    "Carmen Moreno": "F",
    "Carlos Gomez": "M",
    "Maria Santos": "F",
    "Sarah Chen": "F",
    "Mike Johnson": "M",
    "Jennifer Moore": "F",
    "Emily Moore": "F",
    "Diego Moreno": "M",
    "Ana Santos": "F",
    "Dr. Williams": "M",
    "Professor Anderson": "M",
    "Professor Davis": "F",
    "Lisa Park": "F",
    "Mayor Johnson": "M",
    "Miguel Rodriguez": "M",
    "Mrs. Peterson": "F",
    "Officer Thompson": "M",
    "Rachel Kim": "F",
    "Frank Wilson": "M",
}

# Assign voices deterministically — hash the name to spread across available voices
def get_voice(agent_name: str) -> str:
    """Get the KittenTTS voice for an agent."""
    gender = AGENT_GENDER.get(agent_name, "M")
    voices = MALE_VOICES if gender == "M" else FEMALE_VOICES
    idx = int(hashlib.md5(agent_name.encode()).hexdigest(), 16) % len(voices)
    return voices[idx]


# Pre-computed for quick lookup
AGENT_VOICES = {name: get_voice(name) for name in AGENT_GENDER}


async def generate_speech(text: str, agent_name: str, output_path: str,
                          timeout: float = 30.0) -> bool:
    """Generate speech for an agent via KittenTTS endpoint.
    
    Args:
        text: What the agent says
        agent_name: Agent name (for voice selection)
        output_path: Where to save the WAV file
        timeout: Request timeout in seconds
    
    Returns:
        True if successful
    """
    voice = AGENT_VOICES.get(agent_name, "expr-voice-2-m")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                KITTENTTS_URL,
                json={"text": text, "voice": voice, "format": "wav"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"KittenTTS error {resp.status} for {agent_name}")
                    return False
                data = await resp.read()
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(data)
                logger.info(f"Generated speech for {agent_name}: {len(data)} bytes → {output_path}")
                return True
    except Exception as e:
        logger.warning(f"KittenTTS failed for {agent_name}: {e}")
        return False


def generate_speech_sync(text: str, agent_name: str, output_path: str) -> bool:
    """Synchronous wrapper for generate_speech."""
    return asyncio.get_event_loop().run_until_complete(
        generate_speech(text, agent_name, output_path)
    )


if __name__ == "__main__":
    # Print voice assignments
    print("Smallville Agent Voice Map:")
    print("-" * 50)
    for name, voice in sorted(AGENT_VOICES.items()):
        gender = AGENT_GENDER[name]
        print(f"  {name:25s} [{gender}] → {voice}")
