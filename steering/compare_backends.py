#!/usr/bin/env python3
"""
Compare committee (Ollama multi-model) vs steering (single Gemma-2-9B) backends.

Generates outputs for the same agents × scenarios × pipelines from both backends,
then prints side-by-side for qualitative comparison.

Usage:
    python steering/compare_backends.py --steering-only   # test steering first (no Ollama needed)
    python steering/compare_backends.py --both             # compare both (needs Ollama running)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test scenarios — designed to reveal personality differences
SCENARIOS = [
    {
        "name": "Party Invitation",
        "situation": (
            "It's February 14th, Valentine's Day. Isabella Rodriguez has been telling "
            "everyone about a party at Hobbs Cafe from 5-7 PM tonight. You just heard "
            "about it from a neighbor. What do you do?"
        ),
        "pipeline": "decide_action",
    },
    {
        "name": "Greeting at Cafe",
        "situation": (
            "You walk into Hobbs Cafe and see Isabella Rodriguez decorating for the "
            "Valentine's Day party. She smiles and waves at you. What do you say?"
        ),
        "pipeline": "conversation_response",
    },
    {
        "name": "Should Talk?",
        "situation": (
            "You are at Johnson Park and you notice Professor Anderson sitting on a bench "
            "reading a book. You haven't talked to him in a few days."
        ),
        "pipeline": "should_converse",
    },
]

# Test agents — chosen for personality contrast
TEST_AGENTS = [
    "Isabella Rodriguez",   # warm, leader, creative
    "Officer Thompson",     # authoritative, cautious, dutiful
    "Mrs. Peterson",        # nurturing, nostalgic, wise
    "Eddy Lin",            # youthful, curious, warm
    "Frank Wilson",         # practical, trades, reserved
]


async def test_steering():
    """Test steering backend only (no Ollama needed)."""
    os.environ["COMMITTEE_BACKEND"] = "steering"

    # Force re-import
    if "committee" in sys.modules:
        del sys.modules["committee"]

    from committee import PIPELINES, _clean_dialogue, get_committee

    committee = get_committee()
    print("🧬 STEERING BACKEND (Gemma-2-9B + RFM)")
    print("=" * 70)

    results = {}

    for scenario in SCENARIOS:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Pipeline: {scenario['pipeline']}")
        print(f"   {scenario['situation'][:80]}...")
        print("-" * 70)

        for agent in TEST_AGENTS:
            t0 = time.time()
            try:
                raw = await committee.consult(
                    scenario["pipeline"],
                    scenario["situation"],
                    agent,
                )
                if scenario["pipeline"] == "conversation_response":
                    raw = _clean_dialogue(raw, agent)
                elapsed = time.time() - t0
                print(f"\n  🎭 {agent} ({elapsed:.1f}s):")
                print(f"     {raw[:200]}")
                results.setdefault(scenario["name"], {})[agent] = {
                    "output": raw,
                    "time": elapsed,
                    "backend": "steering",
                }
            except Exception as e:
                elapsed = time.time() - t0
                print(f"\n  ❌ {agent} ({elapsed:.1f}s): {e}")
                results.setdefault(scenario["name"], {})[agent] = {
                    "output": f"ERROR: {e}",
                    "time": elapsed,
                    "backend": "steering",
                }

    # Save results
    out_path = Path(__file__).parent / "comparison_results_steering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")

    # Print stats
    stats = committee.get_stats()
    print(f"\n📊 Stats: {stats['total_calls']} total calls")
    for role, count in stats.get("per_expert", {}).items():
        print(f"   {role}: {count}")

    return results


async def test_committee():
    """Test Ollama committee backend."""
    os.environ["COMMITTEE_BACKEND"] = "committee"

    if "committee" in sys.modules:
        del sys.modules["committee"]

    from committee import _clean_dialogue, get_committee

    committee = get_committee()
    print("\n\n🧠 COMMITTEE BACKEND (Ollama multi-model)")
    print("=" * 70)

    results = {}

    for scenario in SCENARIOS:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Pipeline: {scenario['pipeline']}")
        print("-" * 70)

        for agent in TEST_AGENTS:
            t0 = time.time()
            try:
                raw = await committee.consult(
                    scenario["pipeline"],
                    scenario["situation"],
                    agent,
                )
                if scenario["pipeline"] == "conversation_response":
                    raw = _clean_dialogue(raw, agent)
                elapsed = time.time() - t0
                print(f"\n  🎭 {agent} ({elapsed:.1f}s):")
                print(f"     {raw[:200]}")
                results.setdefault(scenario["name"], {})[agent] = {
                    "output": raw,
                    "time": elapsed,
                    "backend": "committee",
                }
            except Exception as e:
                elapsed = time.time() - t0
                print(f"\n  ❌ {agent} ({elapsed:.1f}s): {e}")

    out_path = Path(__file__).parent / "comparison_results_committee.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")

    return results


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--steering-only"

    if mode == "--steering-only":
        await test_steering()
    elif mode == "--both":
        steering_results = await test_steering()
        # Need to unload steering model first to free VRAM
        import gc

        import torch
        if "committee" in sys.modules:
            c = sys.modules["committee"]
            if hasattr(c, '_committee') and c._committee:
                c._committee.shutdown()
                c._committee = None
        gc.collect()
        torch.cuda.empty_cache()
        print("\n⏳ Unloaded steering model, starting Ollama committee test...")
        await test_committee()
    else:
        print(f"Usage: {sys.argv[0]} [--steering-only|--both]")


if __name__ == "__main__":
    asyncio.run(main())
