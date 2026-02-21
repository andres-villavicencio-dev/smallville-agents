"""
Agent personality concept definitions for RFM steering.

Each agent gets a set of concept vectors that steer the base model
toward their personality during inference. Concepts are trained using
the neural_controllers library with ~100 prompt pairs per concept.
"""

# Agent personality profiles — maps agent name to steering concepts and coefficients
# Positive coef = amplify concept, negative = suppress
AGENT_PROFILES = {
    "Isabella Rodriguez": {
        "social_warmth": 0.6,
        "leadership": 0.4,
        "creativity": 0.3,
        "task_focus": -0.2,  # suppress helper tendency
    },
    "Carlos Gomez": {
        "creativity": 0.7,
        "social_warmth": 0.5,
        "enthusiasm": 0.4,
    },
    "Maria Santos": {
        "nurturing": 0.5,
        "social_warmth": 0.4,
        "caution": 0.2,
    },
    "Officer Thompson": {
        "authority": 0.4,
        "caution": 0.5,
        "duty": 0.3,
        "social_warmth": 0.2,
    },
    "Dr. Williams": {
        "analytical": 0.5,
        "expertise_medical": 0.4,
        "empathy": 0.3,
    },
    "Professor Anderson": {
        "academic": 0.6,
        "philosophical": 0.4,
        "mentoring": 0.3,
    },
    "Professor Davis": {
        "academic": 0.5,
        "creativity": 0.3,
        "formality": 0.3,
    },
    "Mrs. Peterson": {
        "nurturing": 0.6,
        "nostalgia": 0.4,
        "social_warmth": 0.5,
        "wisdom": 0.3,
    },
    "Tom Moreno": {
        "practical": 0.6,
        "trades_expertise": 0.4,
        "family_oriented": 0.3,
    },
    "Diego Moreno": {
        "practical": 0.5,
        "youthful_energy": 0.4,
        "family_oriented": 0.3,
    },
    "Carmen Moreno": {
        "organized": 0.5,
        "caution": 0.3,
        "family_oriented": 0.4,
    },
    "Sam Moore": {
        "ambition": 0.5,
        "social_warmth": 0.3,
        "leadership": 0.3,
    },
    "Jennifer Moore": {
        "practical": 0.4,
        "nurturing": 0.3,
        "community_minded": 0.4,
    },
    "Emily Moore": {
        "youthful_energy": 0.5,
        "curiosity": 0.5,
        "social_warmth": 0.3,
    },
    "John Lin": {
        "practical": 0.4,
        "expertise_medical": 0.3,  # pharmacist
        "family_oriented": 0.4,
    },
    "Mei Lin": {
        "academic": 0.4,
        "curiosity": 0.5,
        "family_oriented": 0.3,
    },
    "Eddy Lin": {
        "youthful_energy": 0.6,
        "curiosity": 0.5,
        "social_warmth": 0.3,
    },
    "Mayor Johnson": {
        "authority": 0.4,
        "leadership": 0.5,
        "formality": 0.3,
        "community_minded": 0.5,
    },
    "Rachel Kim": {
        "creativity": 0.6,
        "enthusiasm": 0.4,
        "social_warmth": 0.3,
    },
    "Frank Wilson": {
        "practical": 0.6,
        "trades_expertise": 0.5,
        "reserved": 0.3,
    },
    "Mike Johnson": {
        "reserved": 0.5,
        "practical": 0.4,
        "caution": 0.3,
    },
    "Ana Santos": {
        "creativity": 0.5,
        "social_warmth": 0.4,
        "family_oriented": 0.3,
    },
    "Miguel Rodriguez": {
        "social_warmth": 0.4,
        "practical": 0.3,
        "family_oriented": 0.4,
    },
    "Lisa Park": {
        "organized": 0.5,
        "community_minded": 0.4,
        "social_warmth": 0.3,
    },
    "Sarah Chen": {
        "academic": 0.4,
        "curiosity": 0.5,
        "reserved": 0.2,
    },
}

# Core concepts we need to train RFM directions for
CONCEPTS = [
    "social_warmth",
    "leadership",
    "creativity",
    "enthusiasm",
    "nurturing",
    "caution",
    "authority",
    "duty",
    "analytical",
    "expertise_medical",
    "empathy",
    "academic",
    "philosophical",
    "mentoring",
    "formality",
    "nostalgia",
    "wisdom",
    "practical",
    "trades_expertise",
    "family_oriented",
    "organized",
    "ambition",
    "community_minded",
    "youthful_energy",
    "curiosity",
    "reserved",
    "task_focus",  # to suppress for social reasoning
]


def get_steering_config(agent_name: str) -> dict:
    """Get the base steering coefficients for an agent.
    Returns dict of {concept_name: coefficient} or empty dict if unknown agent."""
    return AGENT_PROFILES.get(agent_name, {})


# ── Pipeline Role Modifiers ──────────────────────────────────────────────────
# Each committee pipeline role adds concept biases on TOP of the agent's base profile.
# This means Isabella's Social expert is warm+creative, while Officer Thompson's
# Social expert is warm+cautious+authoritative.

PIPELINE_ROLE_MODIFIERS = {
    "social": {
        # Social expert should emphasize interpersonal concepts
        "social_warmth": 0.3,
        "empathy": 0.2,
        "task_focus": -0.3,  # suppress task-oriented thinking for social reasoning
    },
    "spatial": {
        # Spatial expert should think practically about locations
        "practical": 0.3,
        "task_focus": 0.1,
    },
    "temporal": {
        # Temporal expert should be organized and practical
        "organized": 0.2,
        "practical": 0.2,
    },
    "emotional": {
        # Emotional expert should emphasize empathy and warmth
        "empathy": 0.4,
        "social_warmth": 0.2,
    },
    "memory": {
        # Memory expert should be analytical and detail-oriented
        "analytical": 0.3,
        "curiosity": 0.2,
    },
    "dialogue": {
        # Dialogue expert should maximize personality distinctiveness
        # (uses full agent profile as-is, with slight creativity boost)
        "creativity": 0.2,
    },
    "judge": {
        # Judge synthesizes — should be balanced and analytical
        "analytical": 0.3,
        "practical": 0.2,
        "authority": 0.1,
    },
}


def get_pipeline_steering(agent_name: str, pipeline_role: str) -> dict:
    """Get composite steering config for a specific agent in a specific pipeline role.

    Merges the agent's base personality with the pipeline role modifiers.
    Agent base + role modifier = personalized expert.

    Example:
        Isabella + social → high warmth, creativity, leadership, suppressed task_focus
        Thompson + social → warmth + caution + authority, moderate task_focus suppression
    """
    base = dict(get_steering_config(agent_name))  # copy
    role_mods = PIPELINE_ROLE_MODIFIERS.get(pipeline_role, {})

    for concept, coef in role_mods.items():
        if concept in base:
            base[concept] = base[concept] + coef  # additive
        else:
            base[concept] = coef

    # Clamp coefficients to [-1.0, 1.0]
    return {k: max(-1.0, min(1.0, v)) for k, v in base.items()}


# ── Example: what each agent's Social expert looks like ──────────────────────
# Isabella Social: {social_warmth: 0.9, leadership: 0.4, creativity: 0.3, task_focus: -0.5, empathy: 0.2}
# Thompson Social: {authority: 0.4, caution: 0.5, duty: 0.3, social_warmth: 0.5, task_focus: -0.3, empathy: 0.2}
# Carlos Social:   {creativity: 0.7, social_warmth: 0.8, enthusiasm: 0.4, task_focus: -0.3, empathy: 0.2}
# Mrs Peterson Social: {nurturing: 0.6, nostalgia: 0.4, social_warmth: 0.8, wisdom: 0.3, task_focus: -0.3, empathy: 0.2}
