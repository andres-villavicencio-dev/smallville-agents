"""
SkillBank — Hierarchical skill library for generative agents.

Inspired by SkillRL (arXiv:2602.08234). Instead of storing raw trajectories,
agents distill experiences into reusable skills that co-evolve over time.

Two skill levels:
  - General: universal strategies that apply across situations
  - Task-specific: specialized knowledge for specific activity types

Skills are distilled from:
  - Successful conversations → social skills
  - Failed conversation attempts → social failure lessons
  - Reflections → cognitive/emotional skills
  - Plan outcomes → planning/spatial skills
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A distilled, reusable behavioral pattern."""
    id: Optional[int] = None
    agent_name: str = ""
    name: str = ""                    # Short label, e.g. "active listening"
    principle: str = ""               # What the skill teaches
    when_to_apply: str = ""           # Conditions for retrieval
    skill_level: str = "task"         # "general" or "task"
    skill_category: str = "social"    # social, planning, emotional, spatial, cognitive
    source_type: str = "conversation" # conversation, reflection, plan, failure
    success: bool = True              # Derived from success or failure?
    use_count: int = 0                # How often retrieved
    effectiveness: float = 0.5        # Running score [0,1] — updated after use
    created_at: datetime = None
    updated_at: datetime = None
    source_description: str = ""      # Original experience it was distilled from

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


class SkillBank:
    """Persistent, evolving skill library per agent."""

    def __init__(self, agent_name: str, db_path: str = "db/memories.db"):
        self.agent_name = agent_name
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self._skill_vectors = None
        self._skill_texts: List[str] = []
        self._skill_ids: List[int] = []
        self._effectiveness_cache: Dict[int, float] = {}  # Cache for effectiveness scores
        self._init_tables()
        self._rebuild_vectors()

    # ── Database ────────────────────────────────────────────────────────────

    def _init_tables(self):
        """Create skills table if it doesn't exist."""
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    principle TEXT NOT NULL,
                    when_to_apply TEXT NOT NULL,
                    skill_level TEXT NOT NULL DEFAULT 'task',
                    skill_category TEXT NOT NULL DEFAULT 'social',
                    source_type TEXT NOT NULL DEFAULT 'conversation',
                    success INTEGER NOT NULL DEFAULT 1,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    effectiveness REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_description TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skills_agent
                ON skills(agent_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skills_category
                ON skills(agent_name, skill_category)
            """)
            conn.commit()

    def add_skill(self, skill: Skill) -> int:
        """Store a new skill. Returns the skill ID."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO skills (
                        agent_name, name, principle, when_to_apply,
                        skill_level, skill_category, source_type, success,
                        use_count, effectiveness, created_at, updated_at,
                        source_description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    skill.agent_name, skill.name, skill.principle,
                    skill.when_to_apply, skill.skill_level, skill.skill_category,
                    skill.source_type, int(skill.success), skill.use_count,
                    skill.effectiveness, skill.created_at.isoformat(),
                    skill.updated_at.isoformat(), skill.source_description
                ))
                conn.commit()
                skill_id = cursor.lastrowid
                self._rebuild_vectors()
                logger.info(f"[{self.agent_name}] New skill: {skill.name} ({skill.skill_category}/{skill.skill_level})")
                return skill_id
        except Exception as e:
            logger.error(f"Error adding skill: {e}")
            return -1

    def get_skills(self, category: Optional[str] = None,
                   level: Optional[str] = None,
                   limit: int = 50) -> List[Skill]:
        """Retrieve skills with optional filters."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                query = "SELECT * FROM skills WHERE agent_name = ?"
                params: list = [self.agent_name]
                if category:
                    query += " AND skill_category = ?"
                    params.append(category)
                if level:
                    query += " AND skill_level = ?"
                    params.append(level)
                query += " ORDER BY effectiveness DESC, use_count DESC LIMIT ?"
                params.append(limit)

                cursor = conn.cursor()
                cursor.execute(query, params)
                return [self._row_to_skill(r) for r in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting skills: {e}")
            return []

    def retrieve_relevant_skills(self, context: str, top_k: int = 5) -> List[Skill]:
        """Retrieve the most relevant skills for a given context using TF-IDF similarity."""
        if self._skill_vectors is None or not self._skill_texts:
            return self.get_skills(limit=top_k)  # Fallback: top by effectiveness

        try:
            query_vec = self.vectorizer.transform([context])
            similarities = cosine_similarity(query_vec, self._skill_vectors)[0]

            # Combine similarity with effectiveness score (use cached effectiveness)
            combined = []
            for i, sim in enumerate(similarities):
                skill_id = self._skill_ids[i]
                # Weighted: 70% relevance, 30% effectiveness
                score = 0.7 * sim + 0.3 * self._get_effectiveness_cached(skill_id)
                combined.append((skill_id, score))

            combined.sort(key=lambda x: x[1], reverse=True)
            top_ids = [sid for sid, _ in combined[:top_k]]

            # Batch fetch all skill objects (single query instead of N)
            skills = self._batch_get_skills_by_ids(top_ids)

            # Batch increment use counts (single query instead of N)
            if top_ids:
                self._batch_increment_use_counts(top_ids)

            return skills

        except Exception as e:
            logger.error(f"Error retrieving relevant skills: {e}")
            return []

    def update_effectiveness(self, skill_id: int, outcome: float):
        """Update a skill's effectiveness with exponential moving average.

        outcome: 1.0 = great result, 0.0 = bad result
        """
        try:
            old = self._effectiveness_cache.get(skill_id, 0.5)
            alpha = 0.3  # EMA weight for new observation
            new_eff = alpha * outcome + (1 - alpha) * old

            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.execute("""
                    UPDATE skills SET effectiveness = ?, updated_at = ? WHERE id = ?
                """, (new_eff, datetime.now().isoformat(), skill_id))
                conn.commit()

            # Update cache
            self._effectiveness_cache[skill_id] = new_eff
        except Exception as e:
            logger.error(f"Error updating effectiveness: {e}")

    def evolve_skills(self, failed_context: str, lesson: str):
        """Recursive evolution: refine or create skills from failure analysis.

        Called periodically (e.g. after a batch of ticks or at end-of-day).
        """
        # Check if an existing skill covers this context
        relevant = self.retrieve_relevant_skills(failed_context, top_k=3)
        for skill in relevant:
            # Penalize existing skills that didn't prevent this failure
            if skill.id:
                self.update_effectiveness(skill.id, 0.2)

        # Create a new failure-lesson skill
        failure_skill = Skill(
            agent_name=self.agent_name,
            name=f"lesson: {lesson[:50]}",
            principle=lesson,
            when_to_apply=failed_context,
            skill_level="task",
            skill_category="cognitive",
            source_type="failure",
            success=False,
            effectiveness=0.6,  # Start slightly above neutral — fresh lessons are valuable
            source_description=failed_context,
        )
        self.add_skill(failure_skill)

    def format_skills_for_prompt(self, skills: List[Skill]) -> str:
        """Format retrieved skills as context for LLM prompts."""
        if not skills:
            return ""
        lines = ["Relevant skills from past experience:"]
        for s in skills:
            tag = "✓" if s.success else "⚠"
            lines.append(f"  {tag} [{s.skill_category}] {s.name}: {s.principle}")
            if not s.success:
                lines.append(f"    When: {s.when_to_apply}")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics for display / logging."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT skill_category, COUNT(*), AVG(effectiveness)
                    FROM skills WHERE agent_name = ?
                    GROUP BY skill_category
                """, (self.agent_name,))
                cats = {r[0]: {"count": r[1], "avg_eff": round(r[2], 2)} for r in cursor.fetchall()}

                cursor.execute("""
                    SELECT COUNT(*) FROM skills WHERE agent_name = ?
                """, (self.agent_name,))
                total = cursor.fetchone()[0]

                return {"total": total, "categories": cats}
        except Exception as e:
            logger.error(f"Error getting skill stats: {e}")
            return {"total": 0, "categories": {}}

    # ── Internal helpers ────────────────────────────────────────────────────

    def _rebuild_vectors(self):
        """Rebuild TF-IDF vectors from all skills."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, principle, when_to_apply, effectiveness
                    FROM skills WHERE agent_name = ?
                """, (self.agent_name,))
                rows = cursor.fetchall()

            if not rows:
                self._skill_vectors = None
                self._skill_texts = []
                self._skill_ids = []
                self._effectiveness_cache = {}
                return

            self._skill_ids = [r[0] for r in rows]
            # Combine name + principle + when_to_apply for richer vectors
            self._skill_texts = [f"{r[1]} {r[2]} {r[3]}" for r in rows]
            self._skill_vectors = self.vectorizer.fit_transform(self._skill_texts)
            # Populate effectiveness cache
            self._effectiveness_cache = {r[0]: r[4] for r in rows}
        except Exception as e:
            logger.error(f"Error rebuilding skill vectors: {e}")
            self._skill_vectors = None

    def _row_to_skill(self, row) -> Skill:
        return Skill(
            id=row[0], agent_name=row[1], name=row[2], principle=row[3],
            when_to_apply=row[4], skill_level=row[5], skill_category=row[6],
            source_type=row[7], success=bool(row[8]), use_count=row[9],
            effectiveness=row[10],
            created_at=datetime.fromisoformat(row[11]),
            updated_at=datetime.fromisoformat(row[12]),
            source_description=row[13],
        )

    def _get_skill_by_id(self, skill_id: int) -> Optional[Skill]:
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM skills WHERE id = ?", (skill_id,))
                row = cursor.fetchone()
                return self._row_to_skill(row) if row else None
        except Exception as e:
            logger.error(f"Error getting skill {skill_id}: {e}")
            return None

    def _get_effectiveness(self, skill_id: int) -> float:
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT effectiveness FROM skills WHERE id = ?", (skill_id,))
                row = cursor.fetchone()
                return row[0] if row else 0.5
        except Exception:
            return 0.5

    def _increment_use_count(self, skill_id: int):
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.execute("""
                    UPDATE skills SET use_count = use_count + 1, updated_at = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), skill_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error incrementing use count: {e}")

    def _get_effectiveness_cached(self, skill_id: int) -> float:
        """Get effectiveness from cache (no DB query)."""
        return self._effectiveness_cache.get(skill_id, 0.5)

    def _batch_get_skills_by_ids(self, skill_ids: List[int]) -> List[Skill]:
        """Fetch multiple skills by ID in a single query."""
        if not skill_ids:
            return []
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                placeholders = ','.join('?' * len(skill_ids))
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM skills WHERE id IN ({placeholders})
                """, skill_ids)
                rows = cursor.fetchall()
                # Preserve order of skill_ids
                skill_map = {row[0]: self._row_to_skill(row) for row in rows}
                return [skill_map[sid] for sid in skill_ids if sid in skill_map]
        except Exception as e:
            logger.error(f"Error batch getting skills: {e}")
            return []

    def _batch_increment_use_counts(self, skill_ids: List[int]):
        """Increment use counts for multiple skills in a single query."""
        if not skill_ids:
            return
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                placeholders = ','.join('?' * len(skill_ids))
                conn.execute(f"""
                    UPDATE skills SET use_count = use_count + 1, updated_at = ?
                    WHERE id IN ({placeholders})
                """, [datetime.now().isoformat()] + skill_ids)
                conn.commit()
        except Exception as e:
            logger.error(f"Error batch incrementing use counts: {e}")


# ── Distillation Functions ──────────────────────────────────────────────────
# These use Ollama to extract skills from raw experience.

async def distill_conversation_skill(
    agent_name: str,
    conversation_text: str,
    outcome: str,  # "success" or "failure"
    location: str = "",
) -> Optional[Skill]:
    """Distill a conversation into a reusable social skill.

    Uses the committee's judge model for distillation.
    """
    from llm import get_llm_client

    success = outcome == "success"
    prompt_type = "successful" if success else "failed/awkward"

    prompt = f"""Analyze this {prompt_type} conversation and extract a reusable social skill.

Agent: {agent_name}
Location: {location}
Conversation:
{conversation_text}

Respond in EXACTLY this format (no extra text):
NAME: <short skill name, 3-6 words>
PRINCIPLE: <what to do/avoid, 1 sentence>
WHEN: <conditions when this applies, 1 sentence>
CATEGORY: <social|emotional|cognitive>"""

    try:
        llm = await get_llm_client()
        response = await llm.generate(prompt, temperature=0.4, max_tokens=150, task="importance")
        return _parse_skill_response(response, agent_name, "conversation", success, conversation_text)
    except Exception as e:
        logger.error(f"Error distilling conversation skill: {e}")
        return None


async def distill_reflection_skill(
    agent_name: str,
    reflection_text: str,
    context: str = "",
) -> Optional[Skill]:
    """Distill a reflection into a cognitive/emotional skill."""
    from llm import get_llm_client

    prompt = f"""Extract a reusable behavioral skill from this agent's reflection.

Agent: {agent_name}
Context: {context}
Reflection: {reflection_text}

Respond in EXACTLY this format (no extra text):
NAME: <short skill name, 3-6 words>
PRINCIPLE: <actionable insight, 1 sentence>
WHEN: <conditions when this applies, 1 sentence>
CATEGORY: <emotional|cognitive|planning|social>"""

    try:
        llm = await get_llm_client()
        response = await llm.generate(prompt, temperature=0.4, max_tokens=150, task="importance")
        return _parse_skill_response(response, agent_name, "reflection", True, reflection_text)
    except Exception as e:
        logger.error(f"Error distilling reflection skill: {e}")
        return None


def _parse_skill_response(
    response: str, agent_name: str, source_type: str,
    success: bool, source_desc: str
) -> Optional[Skill]:
    """Parse the structured LLM response into a Skill object."""
    try:
        lines = response.strip().split("\n")
        parsed = {}
        for line in lines:
            line = line.strip()
            for key in ("NAME:", "PRINCIPLE:", "WHEN:", "CATEGORY:"):
                if line.upper().startswith(key):
                    parsed[key.rstrip(":")] = line[len(key):].strip()
                    break

        name = parsed.get("NAME", "unnamed skill")
        principle = parsed.get("PRINCIPLE", response[:100])
        when = parsed.get("WHEN", "general situations")
        category = parsed.get("CATEGORY", "cognitive").lower()

        if category not in ("social", "emotional", "cognitive", "planning", "spatial"):
            category = "cognitive"

        return Skill(
            agent_name=agent_name,
            name=name,
            principle=principle,
            when_to_apply=when,
            skill_level="task",
            skill_category=category,
            source_type=source_type,
            success=success,
            effectiveness=0.5,
            source_description=source_desc[:500],
        )
    except Exception as e:
        logger.error(f"Error parsing skill response: {e}")
        return None
