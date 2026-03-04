"""Configuration settings for the Generative Agents simulation."""
import os
from typing import Dict, Any

# Model Configuration - Per-task routing for optimal quality/speed
OLLAMA_BASE_URL = "http://localhost:11434"

# Task-specific models (override with env vars)
MODELS = {
    "planning":      os.getenv("MODEL_PLANNING",      "qwen3.5:9b-sim"),   # 9B partial offload: 5.6 GB VRAM, structured plans
    "conversation":  os.getenv("MODEL_CONVERSATION",   "smallville-actor"),  # Fine-tuned 2B character actor
    "reflection":    os.getenv("MODEL_REFLECTION",     "qwen3.5:9b-sim"),   # 9B partial offload: rich reflections
    "importance":    os.getenv("MODEL_IMPORTANCE",     "gemma3:1b"),        # Fast, called hundreds of times
    "default":       os.getenv("OLLAMA_MODEL",         "qwen2.5:3b"),       # Fallback for anything else
}

# Legacy compat
OLLAMA_MODEL = MODELS["default"]

# Committee mode: use mixture-of-experts pipeline instead of single models
# Set USE_COMMITTEE=1 to enable, or pass --committee flag
USE_COMMITTEE = os.getenv("USE_COMMITTEE", "0") == "1"

def is_committee_mode() -> bool:
    """Runtime check for committee mode — survives late flag setting."""
    return USE_COMMITTEE

# Memory Backend Configuration
# Set USE_QDRANT=1 to use semantic vector search instead of TF-IDF
USE_QDRANT = os.getenv("USE_QDRANT", "1") == "1"

# Memory Stream Configuration
MEMORY_RETRIEVAL_WEIGHTS = {
    "recency": 1.0,      # alpha in the paper
    "importance": 1.0,   # beta in the paper  
    "relevance": 1.0     # gamma in the paper
}

# Decay rates
RECENCY_DECAY_FACTOR = 0.99  # Exponential decay for recency scoring
IMPORTANCE_THRESHOLD = 150   # Sum of importance scores to trigger reflection
MAX_RECENT_MEMORIES = 100    # For reflection queries

# Planning Configuration
DAILY_PLAN_CHUNKS = (5, 8)      # Min/max chunks for daily planning
ACTION_DURATION_RANGE = (5, 15)  # Minutes for atomic actions

# Simulation Configuration
DEFAULT_SIMULATION_SPEED = 10    # Game seconds per real second
TICK_DURATION_SECONDS = 180      # Each simulation tick = 3 min game time (480 ticks/day)
DEFAULT_SIM_DAYS = 2            # Default simulation length
DEFAULT_NUM_AGENTS = 25          # Number of agents to simulate (5-25)

# Conversation Configuration
CONVERSATION_PROBABILITY = 0.3   # Chance of initiating conversation when relevant
MAX_CONVERSATION_TURNS = 6       # Maximum turns per conversation
CONVERSATION_RELEVANCE_THRESHOLD = 0.5  # Minimum relevance to initiate

# Performance tuning
AGENT_BATCH_SIZE = 6             # Concurrent agent processing
LLM_SEMAPHORE_LIMIT = 2          # Max concurrent Ollama requests
CONVERSATION_CHECK_INTERVAL = 3  # Only check for new conversations every N ticks
PLANNING_CONCURRENCY = 3         # Max concurrent agent planning calls (Ollama bottleneck)

# Sleep / Curfew Configuration
SLEEP_HARD_START = 23   # 11pm — no new conversations after this hour
SLEEP_HARD_END   = 5    # 5am  — conversations resume after this hour
SLEEP_SOFT_START = 21   # 9pm — start winding down
SLEEP_KEYWORDS = ["sleep", "bed", "rest", "bedroom", "nap", "asleep"]

# Rule-based importance keywords (avoids LLM call per observation)
ROUTINE_IMPORTANCE_KEYWORDS = {
    8: ['party', 'valentine', 'emergency', 'fight', 'love', 'died', 'wedding', 'fired'],
    6: ['invite', 'plan', 'meeting', 'announce', 'new', 'secret'],
    4: ['talk', 'chat', 'discuss', 'mention', 'said'],
    2: ['walk', 'sit', 'stand', 'working', 'eating', 'reading'],
}

# Environment Configuration
SMALLVILLE_LOCATIONS = {
    # Residential — family homes
    "Lin Family Home": ["kitchen", "living_room", "bedroom", "bathroom"],
    "Moreno Family Home": ["kitchen", "living_room", "bedroom", "bathroom"],
    "Moore Family Home": ["kitchen", "living_room", "bedroom", "bathroom"],
    "The Willows": ["apartment_1a", "apartment_2b", "apartment_3c", "hallway"],
    # Residential — individual homes
    "Williams Residence": ["living_room", "bedroom", "study", "kitchen"],
    "Anderson Residence": ["living_room", "bedroom", "study", "kitchen"],
    "Davis Residence": ["living_room", "bedroom", "studio", "kitchen"],
    "Mayor Residence": ["living_room", "bedroom", "study", "kitchen"],
    "Peterson Cottage": ["living_room", "bedroom", "garden", "kitchen"],
    "Thompson Residence": ["living_room", "bedroom", "garage", "kitchen"],
    "Wilson Apartment": ["living_room", "bedroom", "kitchen"],
    "Rodriguez Home": ["living_room", "bedroom", "kitchen", "bathroom"],
    # Public / commercial
    "Oak Hill College": ["classroom_a", "classroom_b", "library", "cafeteria"],
    "Harvey Oak Supply Store": ["front_counter", "aisles", "storage_room"],
    "The Rose and Crown Pub": ["bar_area", "dining_area", "back_room"],
    "Hobbs Cafe": ["counter", "seating_area", "kitchen"],
    "Johnson Park": ["playground", "walking_path", "pond"],
    "Town Hall": ["main_hall", "offices", "meeting_room"],
    "Library": ["reading_area", "computer_lab", "stacks"],
    "Pharmacy": ["counter", "medicine_aisles", "consultation_room"]
}

# Starting scenario configuration
START_DATE = "2023-02-13"  # Day before Valentine's Day
START_TIME = "06:00"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "simulation.log"

def is_hard_sleep_time(sim_hour: int) -> bool:
    """True during hard sleep window — no new conversations allowed."""
    if SLEEP_HARD_START > SLEEP_HARD_END:  # spans midnight
        return sim_hour >= SLEEP_HARD_START or sim_hour < SLEEP_HARD_END
    return SLEEP_HARD_START <= sim_hour < SLEEP_HARD_END

def conversation_sleep_weight(sim_hour: int) -> float:
    """
    Returns a multiplier (0.0–1.0) for conversation probability.
    1.0 = normal, 0.0 = no conversations.
    Drops linearly from SLEEP_SOFT_START to SLEEP_HARD_START.
    """
    if is_hard_sleep_time(sim_hour):
        return 0.0
    if sim_hour < SLEEP_SOFT_START:
        return 1.0
    window = SLEEP_HARD_START - SLEEP_SOFT_START
    elapsed = sim_hour - SLEEP_SOFT_START
    return max(0.0, 1.0 - (elapsed / window))

def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "memory_weights": MEMORY_RETRIEVAL_WEIGHTS,
        "recency_decay": RECENCY_DECAY_FACTOR,
        "importance_threshold": IMPORTANCE_THRESHOLD,
        "simulation_speed": DEFAULT_SIMULATION_SPEED,
        "tick_duration": TICK_DURATION_SECONDS,
        "sim_days": DEFAULT_SIM_DAYS,
        "num_agents": DEFAULT_NUM_AGENTS,
        "locations": SMALLVILLE_LOCATIONS,
        "start_date": START_DATE,
        "start_time": START_TIME,
        "sleep_hard_start": SLEEP_HARD_START,
        "sleep_hard_end": SLEEP_HARD_END,
        "sleep_soft_start": SLEEP_SOFT_START,
    }
