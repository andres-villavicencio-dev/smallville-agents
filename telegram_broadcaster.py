#!/usr/bin/env python3
"""
Smallville Live — Telegram Broadcaster

Tails simulation.log and forwards key events to a Telegram group.
Runs independently from the sim so it can be started/stopped without affecting it.

Usage:
    python3 telegram_broadcaster.py [--dry-run]
"""
import os
import re
import sys
import time
import json
import argparse
import requests
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
TELEGRAM_GROUP = os.getenv("SMALLVILLE_TG_GROUP", "-5210265423")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
LOG_FILE = os.path.join(os.path.dirname(__file__), "simulation.log")
STATE_FILE = os.path.join(os.path.dirname(__file__), ".broadcaster_state.json")

# Rate limiting: don't spam the group
MIN_SEND_INTERVAL = 5.0  # seconds between messages
BATCH_WINDOW = 15.0      # collect events for this long, then send as one message
MAX_BATCH_LINES = 10     # max lines per batched message
DIGEST_INTERVAL = 600    # digest every 10 minutes

# ── What to broadcast ──────────────────────────────────────────────────────
# Each pattern: (regex, emoji, formatter)
# formatter receives the match and returns the broadcast text (or None to skip)

def fmt_conversation_start(m):
    return f"💬 {m.group('a1')} started talking to {m.group('a2')} at {m.group('loc')}"

def fmt_conversation_line(m):
    speaker = m.group('speaker')
    line = m.group('line').strip()
    # Truncate long lines
    if len(line) > 200:
        line = line[:197] + "..."
    return f"🗣 **{speaker}:** {line}"

def fmt_move(m):
    return f"🚶 {m.group('agent')} → {m.group('loc')}"

def fmt_reflection(m):
    agent = m.group('agent')
    reflection = m.group('ref').strip()
    if len(reflection) > 150:
        reflection = reflection[:147] + "..."
    return f"🪞 {agent} reflected: _{reflection}_"

def fmt_skill(m):
    return f"🌱 [{m.group('agent')}] learned: **{m.group('skill')}** ({m.group('cat')})"

def fmt_plan(m):
    return f"📋 {m.group('agent')} planned day with {m.group('count')} activities"

def fmt_converse_check(m):
    result = m.group('result')
    # Only broadcast YES results
    if result == "YES":
        return None  # skip — the "Started conversation" line is more interesting
    return None  # skip NO results too, too noisy

# No real-time broadcasts — everything goes through hourly digest
PATTERNS = []

# Events to suppress (too noisy)
SUPPRESS_PATTERNS = [
    re.compile(r"Checking conversation:"),
    re.compile(r"Conversation check \(committee\)"),
    re.compile(r"should_converse\].*NO"),
    re.compile(r"Error getting"),
    re.compile(r"aiohttp\.access"),
    re.compile(r"WebSocket client"),
]


def load_bot_token():
    """Load bot token from env or openclaw config."""
    global BOT_TOKEN
    if BOT_TOKEN:
        return BOT_TOKEN
    
    config_path = os.path.expanduser("~/.openclaw/openclaw.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        BOT_TOKEN = config.get("channels", {}).get("telegram", {}).get("botToken", "")
    except Exception:
        pass
    return BOT_TOKEN


def send_telegram(text: str, dry_run: bool = False) -> bool:
    """Send a message to the Telegram group."""
    if dry_run:
        print(f"[DRY RUN] {text}")
        return True
    
    token = load_bot_token()
    if not token:
        print("ERROR: No bot token found", file=sys.stderr)
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_GROUP,
        "text": text,
        "parse_mode": "Markdown",
        "disable_notification": True,  # silent by default
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if not resp.ok:
            # Retry without markdown if it fails (markdown can break on special chars)
            payload["parse_mode"] = ""
            payload["text"] = text.replace("**", "").replace("_", "").replace("*", "")
            resp = requests.post(url, json=payload, timeout=10)
        return resp.ok
    except Exception as e:
        print(f"ERROR sending to Telegram: {e}", file=sys.stderr)
        return False


def process_line(line: str) -> str | None:
    """Process a log line and return broadcast text if it's interesting."""
    # Check suppression first
    for pattern in SUPPRESS_PATTERNS:
        if pattern.search(line):
            return None
    
    # Try each pattern
    for pattern, formatter in PATTERNS:
        m = pattern.search(line)
        if m:
            return formatter(m)
    
    return None


# ── Digest patterns (collected hourly, not broadcast immediately) ───────
RE_REFLECTION = re.compile(r"(?P<agent>\w[\w ]+?) reflected \(committee\): (?P<ref>.+)$")
RE_SKILL = re.compile(r"\[(?P<agent>.+?)\] New skill: (?P<skill>.+?) \((?P<cat>.+?)\)$")
RE_PLAN = re.compile(r"(?P<agent>\w[\w ]+?) planned day with (?P<count>\d+) activities$")
RE_MOVE = re.compile(r"environment.*?(?P<agent>\w[\w ]+?) moved to (?P<loc>.+)$")
RE_CONV_START = re.compile(r"Started conversation between (?P<a1>.+?) and (?P<a2>.+?) at (?P<loc>.+)$")
RE_CONV_LINE = re.compile(r"Conversation: (?P<speaker>.+?): (?P<line>.+)$")


def collect_digest_event(line, reflections, skills, plans, moves, conversations=None):
    """Collect events for the hourly digest."""
    m = RE_REFLECTION.search(line)
    if m:
        ref = m.group('ref').strip()
        if len(ref) > 80:
            ref = ref[:77] + "..."
        reflections.append((m.group('agent'), ref))
        return
    
    m = RE_SKILL.search(line)
    if m:
        skills.append((m.group('agent'), m.group('skill'), m.group('cat')))
        return
    
    m = RE_PLAN.search(line)
    if m:
        plans.append((m.group('agent'), m.group('count')))
        return
    
    m = RE_MOVE.search(line)
    if m:
        moves[m.group('agent')] = m.group('loc')
        return
    
    if conversations is not None:
        m = RE_CONV_START.search(line)
        if m:
            key = f"{m.group('a1')} & {m.group('a2')}"
            conversations.setdefault(key, {"loc": m.group('loc'), "lines": []})
            return
        
        m = RE_CONV_LINE.search(line)
        if m:
            # Attach to the most recent conversation involving this speaker
            speaker = m.group('speaker')
            msg = m.group('line').strip()
            for key in reversed(list(conversations.keys())):
                if speaker in key:
                    conversations[key]["lines"].append((speaker, msg))
                    break


def build_digest(reflections, skills, plans, moves, conversations=None):
    """Build the hourly digest message. Returns None if nothing happened."""
    conversations = conversations or {}
    if not reflections and not skills and not plans and not moves and not conversations:
        return None
    
    parts = ["📊 **Smallville Hourly Digest**\n"]
    
    if conversations:
        parts.append("💬 **Conversations:**")
        for pair, data in conversations.items():
            loc = data["loc"]
            lines = data["lines"]
            if not lines:
                parts.append(f"  • {pair} at {loc} (no dialogue captured)")
                continue
            # Build a short summary: first line + topic hint from key phrases
            topics = _summarize_conversation(lines)
            parts.append(f"  • {pair} at {loc} ({len(lines)} turns)")
            parts.append(f"    _{topics}_")
        parts.append("")
    
    if moves:
        parts.append("🚶 **Current Locations:**")
        loc_agents = {}
        for agent, loc in moves.items():
            loc_agents.setdefault(loc, []).append(agent)
        for loc, agents in sorted(loc_agents.items()):
            parts.append(f"  • {loc}: {', '.join(agents)}")
        parts.append("")
    
    if reflections:
        agent_refs = {}
        for agent, ref in reflections:
            agent_refs[agent] = ref
        parts.append("🪞 **Reflections:**")
        for agent, ref in agent_refs.items():
            parts.append(f"  • {agent}: _{ref}_")
        parts.append("")
    
    if skills:
        agent_skills = {}
        for agent, skill, cat in skills:
            agent_skills.setdefault(agent, []).append(skill)
        parts.append("🌱 **Skills Learned:**")
        for agent, skill_list in agent_skills.items():
            parts.append(f"  • {agent}: {', '.join(skill_list)}")
    
    if plans:
        parts.append("")
        parts.append("📋 **Plans:**")
        for agent, count in plans:
            parts.append(f"  • {agent}: {count} activities")
    
    return "\n".join(parts)


def _summarize_conversation(lines):
    """Extract a brief summary from conversation lines."""
    # Take first and last line to capture opening topic + conclusion
    all_text = " ".join(msg for _, msg in lines)
    # Grab first sentence of first speaker and last speaker as summary
    first_msg = lines[0][1]
    last_msg = lines[-1][1] if len(lines) > 1 else ""
    
    # Truncate each to ~60 chars
    first_short = first_msg[:60] + ("..." if len(first_msg) > 60 else "")
    
    if last_msg and len(lines) > 2:
        last_short = last_msg[:60] + ("..." if len(last_msg) > 60 else "")
        return f"{first_short} → {last_short}"
    else:
        return first_short


def load_state() -> int:
    """Load the last processed byte offset."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f).get("offset", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def save_state(offset: int):
    """Save the current byte offset."""
    with open(STATE_FILE, "w") as f:
        json.dump({"offset": offset, "ts": datetime.now().isoformat()}, f)


def tail_and_broadcast(dry_run: bool = False):
    """Main loop: tail the log file and broadcast events."""
    print(f"📡 Smallville Live Broadcaster")
    print(f"   Log: {LOG_FILE}")
    print(f"   Group: {TELEGRAM_GROUP}")
    print(f"   Dry run: {dry_run}")
    print(f"   Batch window: {BATCH_WINDOW}s")
    print(f"   Digest interval: {DIGEST_INTERVAL}s")
    print()
    
    offset = load_state()
    
    # If log file is smaller than offset, it was rotated/recreated
    try:
        file_size = os.path.getsize(LOG_FILE)
        if file_size < offset:
            print(f"Log file rotated (size {file_size} < offset {offset}), starting from beginning")
            offset = 0
    except FileNotFoundError:
        print("Waiting for simulation.log to appear...")
        while not os.path.exists(LOG_FILE):
            time.sleep(2)
        offset = 0
    
    # Start from end of file if no saved state (don't replay old events)
    if offset == 0 and os.path.exists(LOG_FILE):
        offset = os.path.getsize(LOG_FILE)
        print(f"Starting from end of file (offset {offset})")
    
    last_send_time = 0
    batch = []
    
    # Hourly digest accumulators
    last_digest_time = time.time()
    digest_reflections = []   # (agent, reflection_text)
    digest_skills = []        # (agent, skill_name, category)
    digest_plans = []         # (agent, count)
    digest_moves = {}         # agent -> last location (only track latest)
    digest_conversations = {} # "A & B" -> {loc, lines}
    
    while True:
        try:
            with open(LOG_FILE, "r") as f:
                f.seek(offset)
                new_lines = f.readlines()
                offset = f.tell()
            
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for digest events (reflections, skills, plans, moves, conversations)
                collect_digest_event(line, digest_reflections, digest_skills, digest_plans, digest_moves, digest_conversations)
                
                broadcast_text = process_line(line)
                if broadcast_text:
                    batch.append(broadcast_text)
            
            now = time.time()
            
            # Send conversation batch if ready
            if batch and (now - last_send_time >= BATCH_WINDOW or len(batch) >= MAX_BATCH_LINES):
                message = "\n".join(batch[:MAX_BATCH_LINES])
                if len(batch) > MAX_BATCH_LINES:
                    message += f"\n... and {len(batch) - MAX_BATCH_LINES} more events"
                
                send_telegram(message, dry_run)
                last_send_time = now
                batch.clear()
                save_state(offset)
            
            # Send hourly digest if due
            if now - last_digest_time >= DIGEST_INTERVAL:
                digest_msg = build_digest(digest_reflections, digest_skills, digest_plans, digest_moves, digest_conversations)
                if digest_msg:
                    send_telegram(digest_msg, dry_run)
                    last_send_time = now
                # Reset accumulators
                digest_reflections.clear()
                digest_skills.clear()
                digest_plans.clear()
                digest_moves.clear()
                digest_conversations.clear()
                last_digest_time = now
                save_state(offset)
            
            time.sleep(1.0)
            
        except KeyboardInterrupt:
            print("\nShutting down broadcaster")
            save_state(offset)
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smallville Live Telegram Broadcaster")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of sending")
    args = parser.parse_args()
    
    tail_and_broadcast(dry_run=args.dry_run)
