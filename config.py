"""Configuration settings for the Generative Agents simulation."""
import os
from typing import Dict, Any

# Model Configuration - Per-task routing for optimal quality/speed
OLLAMA_BASE_URL = "http://localhost:11434"

# Task-specific models (override with env vars)
MODELS = {
    "planning":      os.getenv("MODEL_PLANNING",      "qwen2.5:3b"),      # Structured output, daily plans
    "conversation":  os.getenv("MODEL_CONVERSATION",   "llama3.2:3b"),     # Chat-tuned, natural dialogue
    "reflection":    os.getenv("MODEL_REFLECTION",     "gemma3:4b"),       # Synthesis & abstract reasoning
    "importance":    os.getenv("MODEL_IMPORTANCE",     "gemma3:1b"),       # Fast, called hundreds of times
    "default":       os.getenv("OLLAMA_MODEL",         "qwen2.5:3b"),      # Fallback for anything else
}

# Legacy compat
OLLAMA_MODEL = MODELS["default"]

# Committee mode: use mixture-of-experts pipeline instead of single models
# Set USE_COMMITTEE=1 to enable, or pass --committee flag
USE_COMMITTEE = os.getenv("USE_COMMITTEE", "0") == "1"

def is_committee_mode() -> bool:
    """Runtime check for committee mode — survives late flag setting."""
    return USE_COMMITTEE

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
TICK_DURATION_SECONDS = 10       # Each simulation tick = 10 game seconds
DEFAULT_SIM_DAYS = 2            # Default simulation length
DEFAULT_NUM_AGENTS = 25          # Number of agents to simulate (5-25)

# Conversation Configuration
CONVERSATION_PROBABILITY = 0.3   # Chance of initiating conversation when relevant
MAX_CONVERSATION_TURNS = 6       # Maximum turns per conversation
CONVERSATION_RELEVANCE_THRESHOLD = 0.5  # Minimum relevance to initiate

# Environment Configuration
SMALLVILLE_LOCATIONS = {
    "Lin Family Home": ["kitchen", "living_room", "bedroom", "bathroom"],
    "Moreno Family Home": ["kitchen", "living_room", "bedroom", "bathroom"],
    "Moore Family Home": ["kitchen", "living_room", "bedroom", "bathroom"],
    "The Willows": ["apartment_1a", "apartment_2b", "apartment_3c", "hallway"],
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
START_TIME = "08:00"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "simulation.log"

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
        "start_time": START_TIME
    }