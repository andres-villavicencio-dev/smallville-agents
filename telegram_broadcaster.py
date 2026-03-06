#!/usr/bin/env python3
"""
Smallville Live — Telegram Broadcaster

Tails simulation.log and sends rich status digests to a Telegram group.
Digest format: location clusters, active conversations, re-plans, 
Isabella party tracking, reflection/skill counts.

Usage:
    python3 telegram_broadcaster.py [--dry-run]
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

# ── Config ──────────────────────────────────────────────────────────────────
TELEGRAM_GROUP = os.getenv("SMALLVILLE_TG_GROUP", "-5210265423")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
LOG_FILE = os.path.join(os.path.dirname(__file__), "simulation.log")
STATE_FILE = os.path.join(os.path.dirname(__file__), ".broadcaster_state.json")
DIGEST_INTERVAL = 600  # 10 minutes


# ── Regex patterns ──────────────────────────────────────────────────────────
RE_MOVE = re.compile(r"environment.*?(?P<agent>\w[\w ]+?) moved to (?P<loc>.+)$")
RE_CONV_START = re.compile(r"Started conversation between (?P<a1>.+?) and (?P<a2>.+?) at (?P<loc>.+)$")
RE_CONV_END = re.compile(r"Ended conversation between (?P<a1>.+?) and (?P<a2>.+)$")
RE_CONV_LINE = re.compile(r"Conversation: (?P<speaker>.+?): (?P<line>.+)$")
RE_REFLECTION = re.compile(r"(?P<agent>\w[\w ]+?) reflected \(committee\): (?P<ref>.+)$")
RE_SKILL = re.compile(r"\[(?P<agent>.+?)\] New skill: (?P<skill>.+?) \((?P<cat>.+?)\)$")
RE_REPLAN = re.compile(r"\[replan\] (?P<agent>.+?) added: (?P<time>\S+) — (?P<desc>.+)$")
RE_PLAN = re.compile(r"(?P<agent>\w[\w ]+?) planned day with (?P<count>\d+) activities$")
RE_PARTY = re.compile(r"(?i)(valentine|party|hobbs cafe.*party|party.*hobbs)")
RE_SIM_TIME = re.compile(r"time (?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_TICK = re.compile(r"tick (?P<tick>\d+), time (?P<time>.+)$")


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
        print(f"[DRY RUN]\n{text}\n")
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
        "disable_notification": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if not resp.ok:
            # Retry without markdown
            payload["parse_mode"] = ""
            payload["text"] = text.replace("**", "").replace("_", "").replace("*", "")
            resp = requests.post(url, json=payload, timeout=10)
        return resp.ok
    except Exception as e:
        print(f"ERROR sending to Telegram: {e}", file=sys.stderr)
        return False


class DigestCollector:
    """Collects events between digest intervals."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.locations = {}           # agent -> latest location
        self.conversations = []       # [(a1, a2, loc, opening_line)]
        self.conv_active = {}         # "a1 & a2" -> {loc, lines, opening}
        self.replans = []             # [(agent, desc)]
        self.reflections = defaultdict(int)  # agent -> count
        self.skills = defaultdict(set)       # agent -> {skill names}
        self.party_mentions = []      # [(agent, line)]
        self.plans = []               # [(agent, count)]

    def process_line(self, line: str):
        """Process a log line and collect events."""

        m = RE_MOVE.search(line)
        if m:
            self.locations[m.group('agent')] = m.group('loc')
            return

        m = RE_CONV_START.search(line)
        if m:
            key = f"{m.group('a1')} & {m.group('a2')}"
            self.conv_active[key] = {
                "loc": m.group('loc'),
                "a1": m.group('a1'),
                "a2": m.group('a2'),
                "opening": None,
                "turns": 0,
            }
            return

        m = RE_CONV_LINE.search(line)
        if m:
            speaker = m.group('speaker')
            msg = m.group('line').strip()
            # Attach to active conversation
            for key, conv in self.conv_active.items():
                if speaker in key:
                    conv["turns"] += 1
                    if conv["opening"] is None:
                        conv["opening"] = msg
                    # Track party mentions
                    if RE_PARTY.search(msg):
                        self.party_mentions.append((speaker, msg))
                    break
            return

        m = RE_CONV_END.search(line)
        if m:
            key1 = f"{m.group('a1')} & {m.group('a2')}"
            key2 = f"{m.group('a2')} & {m.group('a1')}"
            conv = self.conv_active.pop(key1, None) or self.conv_active.pop(key2, None)
            if conv and conv["turns"] > 0:
                opening = conv["opening"] or ""
                if len(opening) > 80:
                    opening = opening[:77] + "..."
                self.conversations.append((
                    conv["a1"], conv["a2"], conv["loc"],
                    opening, conv["turns"]
                ))
            return

        m = RE_REPLAN.search(line)
        if m:
            desc = m.group('desc')
            if len(desc) > 100:
                desc = desc[:97] + "..."
            self.replans.append((m.group('agent'), f"{m.group('time')} — {desc}"))
            return

        m = RE_REFLECTION.search(line)
        if m:
            self.reflections[m.group('agent')] += 1
            return

        m = RE_SKILL.search(line)
        if m:
            self.skills[m.group('agent')].add(m.group('skill'))
            return

        m = RE_PLAN.search(line)
        if m:
            self.plans.append((m.group('agent'), m.group('count')))
            return

    def build_digest(self) -> str | None:
        """Build a rich status digest."""
        has_content = (self.conversations or self.replans or self.locations
                       or self.reflections or self.skills or self.party_mentions)
        if not has_content:
            return None

        # Read sim time from state file
        sim_time_str = ""
        try:
            state_path = os.path.join(os.path.dirname(__file__), "saves", "latest_state.json")
            with open(state_path) as f:
                state = json.load(f)
            sim_dt = datetime.fromisoformat(state.get("current_time", ""))
            sim_time_str = sim_dt.strftime("%I:%M %p, %b %d")
            tick = state.get("tick_count", "?")
        except Exception:
            sim_time_str = ""
            tick = "?"

        header = "🏘 *Smallville Live*"
        if sim_time_str:
            header += f" — _{sim_time_str}_ (tick {tick})"
        parts = [header + "\n"]

        # ── Location clusters ──
        if self.locations:
            loc_agents = defaultdict(list)
            for agent, loc in self.locations.items():
                loc_agents[loc].append(agent.split()[0])
            # Sort by agent count, show all locations with agents
            sorted_locs = sorted(loc_agents.items(), key=lambda x: -len(x[1]))
            loc_lines = []
            for loc, agents in sorted_locs:
                if len(agents) >= 2:
                    loc_lines.append(f"  *{loc}* — {', '.join(agents)} ({len(agents)})")
                else:
                    loc_lines.append(f"  {loc} — {agents[0]}")
            if loc_lines:
                parts.append("📍 *Where everyone is:*")
                parts.extend(loc_lines)
                parts.append("")

        # ── Party cascade tracking ──
        if self.party_mentions:
            parts.append("🎉 *Valentine's Day Party Mentions:*")
            seen = set()
            for speaker, msg in self.party_mentions:
                if speaker not in seen:
                    snippet = msg[:80] + ("..." if len(msg) > 80 else "")
                    parts.append(f"  {speaker}: _{snippet}_")
                    seen.add(speaker)
            parts.append("")

        # ── Re-plans (most interesting!) ──
        if self.replans:
            parts.append("🔄 *Re-plans triggered:*")
            for agent, desc in self.replans:
                parts.append(f"  {agent} → {desc}")
            parts.append("")

        # ── Conversations (compact summary) ──
        if self.conversations:
            # Group by location
            conv_by_loc = defaultdict(list)
            for a1, a2, loc, opening, turns in self.conversations:
                conv_by_loc[loc].append((a1.split()[0], a2.split()[0], turns))
            parts.append(f"💬 *{len(self.conversations)} conversations ended:*")
            for loc, convos in conv_by_loc.items():
                pairs = ", ".join(f"{a}&{b} ({t}t)" for a, b, t in convos)
                parts.append(f"  {loc}: {pairs}")
            parts.append("")

        # ── Active conversations (brief) ──
        if self.conv_active:
            active_by_loc = defaultdict(list)
            for key, conv in self.conv_active.items():
                active_by_loc[conv["loc"]].append(
                    f"{conv['a1'].split()[0]}&{conv['a2'].split()[0]}"
                )
            active_parts = []
            for loc, pairs in active_by_loc.items():
                active_parts.append(f"{loc}: {', '.join(pairs)}")
            parts.append(f"🗣 *{len(self.conv_active)} active:* {'; '.join(active_parts)}")
            parts.append("")

        # ── Reflections ──
        if self.reflections:
            total = sum(self.reflections.values())
            top = sorted(self.reflections.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{a.split()[0]} ({c})" for a, c in top)
            parts.append(f"🪞 {total} reflections — {top_str}")

        # ── Skills ──
        if self.skills:
            total = sum(len(s) for s in self.skills.values())
            top = sorted(self.skills.items(), key=lambda x: -len(x[1]))[:3]
            top_str = ", ".join(f"{a.split()[0]} ({len(s)})" for a, s in top)
            parts.append(f"🌱 {total} new skills — {top_str}")

        return "\n".join(parts)


def load_state() -> int:
    try:
        with open(STATE_FILE) as f:
            return json.load(f).get("offset", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def save_state(offset: int):
    with open(STATE_FILE, "w") as f:
        json.dump({"offset": offset, "ts": datetime.now().isoformat()}, f)


def tail_and_broadcast(dry_run: bool = False):
    """Main loop: tail log, collect events, send periodic digests."""
    print("📡 Smallville Live Broadcaster")
    print(f"   Log: {LOG_FILE}")
    print(f"   Group: {TELEGRAM_GROUP}")
    print(f"   Digest every: {DIGEST_INTERVAL}s")
    print(f"   Dry run: {dry_run}")
    print()

    offset = load_state()

    # Handle log rotation
    try:
        file_size = os.path.getsize(LOG_FILE)
        if file_size < offset:
            offset = 0
    except FileNotFoundError:
        print("Waiting for simulation.log...")
        while not os.path.exists(LOG_FILE):
            time.sleep(2)
        offset = 0

    # Start from end if no saved state
    if offset == 0 and os.path.exists(LOG_FILE):
        offset = os.path.getsize(LOG_FILE)
        print(f"Starting from end of file (offset {offset})")

    collector = DigestCollector()
    last_digest_time = time.time()

    while True:
        try:
            with open(LOG_FILE) as f:
                f.seek(offset)
                new_lines = f.readlines()
                offset = f.tell()

            for line in new_lines:
                line = line.strip()
                if line:
                    collector.process_line(line)

            now = time.time()

            # Send digest if due
            if now - last_digest_time >= DIGEST_INTERVAL:
                digest = collector.build_digest()
                if digest:
                    send_telegram(digest, dry_run)
                collector.reset()
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
