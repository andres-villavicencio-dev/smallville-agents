"""Voice integration for Smallville — generates speech for conversations via KittenTTS.

Hooks into the conversation system to generate audio files for each turn.
Can also produce a merged conversation audio clip for the Telegram broadcaster.
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

KITTENTTS_URL = os.getenv("KITTENTTS_URL", "http://192.168.1.70:8377/tts")
VOICE_DIR = os.path.join(os.path.dirname(__file__), "voices")
ENABLED = os.getenv("SMALLVILLE_VOICES", "1") == "1"

# Import voice mapping
from voice_map import AGENT_VOICES


async def generate_turn_audio(speaker: str, text: str, output_path: str,
                               timeout: float = 30.0) -> bool:
    """Generate audio for a single conversation turn."""
    if not ENABLED:
        return False
    
    voice = AGENT_VOICES.get(speaker, "expr-voice-2-m")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                KITTENTTS_URL,
                json={"text": text, "voice": voice, "format": "wav"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"KittenTTS error {resp.status} for {speaker}")
                    return False
                data = await resp.read()
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(data)
                return True
    except Exception as e:
        logger.warning(f"KittenTTS request failed for {speaker}: {e}")
        return False


async def generate_conversation_audio(
    turns: list[tuple[str, str]],  # [(speaker, message), ...]
    conversation_id: str,
) -> str | None:
    """Generate audio for an entire conversation, returning path to merged OGG.
    
    Each turn is synthesized separately (with the agent's voice), then
    concatenated with a short silence between turns and converted to OGG.
    """
    if not ENABLED or not turns:
        return None
    
    conv_dir = os.path.join(VOICE_DIR, conversation_id)
    os.makedirs(conv_dir, exist_ok=True)
    
    turn_files = []
    for i, (speaker, message) in enumerate(turns):
        wav_path = os.path.join(conv_dir, f"turn_{i:02d}.wav")
        ok = await generate_turn_audio(speaker, message, wav_path)
        if ok:
            turn_files.append(wav_path)
        else:
            logger.warning(f"Skipping turn {i} ({speaker}) — TTS failed")
    
    if not turn_files:
        return None
    
    # Concatenate with 0.5s silence between turns using ffmpeg
    output_ogg = os.path.join(VOICE_DIR, f"{conversation_id}.ogg")
    
    try:
        # Build ffmpeg filter to concatenate with silence gaps
        inputs = []
        filter_parts = []
        
        for i, f in enumerate(turn_files):
            inputs.extend(["-i", f])
            filter_parts.append(f"[{i}:a]")
        
        # Add silence between turns
        n = len(turn_files)
        filter_str = "".join(filter_parts) + f"concat=n={n}:v=0:a=1[out]"
        
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[out]",
            "-c:a", "libopus", "-b:a", "64k",
            output_ogg,
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            logger.warning(f"ffmpeg concat failed: {result.stderr.decode()[-200:]}")
            return None
        
        logger.info(f"Conversation audio: {output_ogg} ({os.path.getsize(output_ogg)} bytes)")
        
        # Clean up individual turn files
        for f in turn_files:
            os.remove(f)
        try:
            os.rmdir(conv_dir)
        except OSError:
            pass
        
        return output_ogg
        
    except Exception as e:
        logger.warning(f"Failed to merge conversation audio: {e}")
        return None


async def voice_conversation_end(conversation) -> str | None:
    """Hook called when a conversation ends. Returns path to OGG if successful."""
    if not ENABLED:
        return None
    
    turns = [(t.speaker, t.message) for t in conversation.turns]
    if not turns:
        return None
    
    # Use agent names as conversation ID
    conv_id = f"{conversation.agent1}_{conversation.agent2}_{conversation.start_time.strftime('%H%M%S')}"
    conv_id = conv_id.replace(" ", "_").replace(".", "")
    
    return await generate_conversation_audio(turns, conv_id)
