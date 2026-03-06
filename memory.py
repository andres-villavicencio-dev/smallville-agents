"""Memory stream implementation for generative agents."""
import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import MEMORY_RETRIEVAL_WEIGHTS, RECENCY_DECAY_FACTOR

logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """A single memory object."""
    id: int | None = None
    agent_name: str = ""
    description: str = ""
    memory_type: str = "observation"  # observation, reflection, plan
    importance_score: int = 5
    creation_timestamp: datetime = None
    last_access_timestamp: datetime = None
    location: str = ""
    source_memory_ids: list[int] = None  # For reflections
    
    def __post_init__(self):
        if self.creation_timestamp is None:
            self.creation_timestamp = datetime.now()
        if self.last_access_timestamp is None:
            self.last_access_timestamp = self.creation_timestamp
        if self.source_memory_ids is None:
            self.source_memory_ids = []
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['creation_timestamp'] = self.creation_timestamp.isoformat()
        data['last_access_timestamp'] = self.last_access_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Memory':
        """Create from dictionary."""
        data['creation_timestamp'] = datetime.fromisoformat(data['creation_timestamp'])
        data['last_access_timestamp'] = datetime.fromisoformat(data['last_access_timestamp'])
        return cls(**data)

class MemoryStream:
    """Manages an agent's memory stream with retrieval and scoring."""
    
    def __init__(self, agent_name: str, db_path: str = "db/memories.db"):
        self.agent_name = agent_name
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.memory_vectors = None
        self.memory_descriptions = []
        self._vector_cache_valid = False  # Track if vectors need rebuild
        self._init_database()
        self._load_memories_for_vectorization()
    
    def _init_database(self):
        """Initialize the SQLite database with tables and triggers."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

                # Create main memories table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        importance_score INTEGER NOT NULL,
                        creation_timestamp TEXT NOT NULL,
                        last_access_timestamp TEXT NOT NULL,
                        location TEXT,
                        source_memory_ids TEXT
                    )
                """)

                # Create FTS5 table for full-text search
                self._create_fts_table(conn)

                # Create triggers to keep FTS in sync
                self._create_fts_triggers(conn)

                # Create performance indexes for common queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_agent_timestamp
                    ON memories(agent_name, creation_timestamp DESC)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_agent_type
                    ON memories(agent_name, memory_type)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                    ON memories(creation_timestamp DESC)
                """)

                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def _create_fts_table(self, conn):
        """Create FTS5 virtual table for full-text search."""
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id UNINDEXED,
                description
            )
        """)

    def _create_fts_triggers(self, conn):
        """Create triggers to keep FTS table synchronized with main table."""
        # Trigger on INSERT
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(memory_id, description) VALUES (new.id, new.description);
            END
        """)

        # Trigger on DELETE
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, memory_id, description)
                VALUES ('delete', old.id, old.description);
            END
        """)

        # Trigger on UPDATE
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF description ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, memory_id, description)
                VALUES ('delete', old.id, old.description);
                INSERT INTO memories_fts(memory_id, description) VALUES (new.id, new.description);
            END
        """)
    
    def add_memory(self, memory: Memory) -> int:
        """Add a memory to the stream."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memories (
                        agent_name, description, memory_type, importance_score,
                        creation_timestamp, last_access_timestamp, location, source_memory_ids
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.agent_name,
                    memory.description,
                    memory.memory_type,
                    memory.importance_score,
                    memory.creation_timestamp.isoformat(),
                    memory.last_access_timestamp.isoformat(),
                    memory.location,
                    json.dumps(memory.source_memory_ids)
                ))
                memory_id = cursor.lastrowid
                conn.commit()

                # Invalidate vector cache (will rebuild on next retrieval)
                self._vector_cache_valid = False

                return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return -1
    
    def get_memories(self, limit: int | None = None, 
                    memory_type: str | None = None) -> list[Memory]:
        """Get memories for the agent."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                query = "SELECT * FROM memories WHERE agent_name = ?"
                params = [self.agent_name]
                
                if memory_type:
                    query += " AND memory_type = ?"
                    params.append(memory_type)
                
                query += " ORDER BY creation_timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_memory(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting memories: {e}")
            return []

    @staticmethod
    def _row_to_memory(row) -> Memory:
        """Convert a database row tuple to a Memory object."""
        return Memory(
            id=row[0],
            agent_name=row[1],
            description=row[2],
            memory_type=row[3],
            importance_score=row[4],
            creation_timestamp=datetime.fromisoformat(row[5]),
            last_access_timestamp=datetime.fromisoformat(row[6]),
            location=row[7] or "",
            source_memory_ids=json.loads(row[8]) if row[8] else []
        )

    def _load_memories_for_vectorization(self):
        """Load all memories for TF-IDF vectorization."""
        memories = self.get_memories()
        if not memories:
            self.memory_descriptions = []
            self.memory_vectors = None
            self._vector_cache_valid = True
            return

        self.memory_descriptions = [m.description for m in memories]
        try:
            self.memory_vectors = self.vectorizer.fit_transform(self.memory_descriptions)
            self._vector_cache_valid = True
        except Exception as e:
            logger.warning(f"Error creating memory vectors: {e}")
            self.memory_vectors = None
            self._vector_cache_valid = False
    
    def _calculate_recency_score(self, memory: Memory, current_time: datetime) -> float:
        """Calculate recency score using exponential decay."""
        hours_since_access = (current_time - memory.last_access_timestamp).total_seconds() / 3600
        return math.pow(RECENCY_DECAY_FACTOR, hours_since_access)
    
    def _calculate_importance_score(self, memory: Memory) -> float:
        """Normalize importance score to [0, 1]."""
        return (memory.importance_score - 1) / 9.0  # 1-10 scale to 0-1
    
    def retrieve_memories(self, query: str, top_k: int = 10,
                         current_time: datetime | None = None) -> list[tuple[Memory, float]]:
        """Retrieve top-k most relevant memories with combined scoring."""
        if current_time is None:
            current_time = datetime.now()

        memories = self.get_memories()
        if not memories:
            return []

        # Rebuild vector cache if invalid
        if not self._vector_cache_valid:
            self._load_memories_for_vectorization()

        # Batch-compute relevance scores (1 vectorization op, not N)
        relevance_scores = {}
        if self.memory_vectors is not None and query.strip():
            try:
                query_vector = self.vectorizer.transform([query])
                all_similarities = cosine_similarity(query_vector, self.memory_vectors)[0]
                for i, desc in enumerate(self.memory_descriptions):
                    if i < len(all_similarities):
                        relevance_scores[desc] = float(all_similarities[i])
            except Exception:
                pass

        memory_scores = []
        for memory in memories:
            memory.last_access_timestamp = current_time

            recency = self._calculate_recency_score(memory, current_time)
            importance = self._calculate_importance_score(memory)
            relevance = relevance_scores.get(memory.description, 0.5)

            weights = MEMORY_RETRIEVAL_WEIGHTS
            combined_score = (
                weights["recency"] * recency +
                weights["importance"] * importance +
                weights["relevance"] * relevance
            ) / sum(weights.values())

            memory_scores.append((memory, combined_score))

        memory_scores.sort(key=lambda x: x[1], reverse=True)
        top_memories = memory_scores[:top_k]

        # Batch update access timestamps (single query instead of N)
        if top_memories:
            memory_ids = [m.id for m, _ in top_memories]
            self._batch_update_memory_access_times(memory_ids, current_time)

        return top_memories
    
    def _update_memory_access_time(self, memory_id: int, access_time: datetime):
        """Update the last access time for a memory."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.execute("""
                    UPDATE memories
                    SET last_access_timestamp = ?
                    WHERE id = ?
                """, (access_time.isoformat(), memory_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating memory access time: {e}")

    def _batch_update_memory_access_times(self, memory_ids: list[int], access_time: datetime):
        """Batch update last access times for multiple memories (single query)."""
        if not memory_ids:
            return
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                placeholders = ','.join('?' * len(memory_ids))
                conn.execute(f"""
                    UPDATE memories
                    SET last_access_timestamp = ?
                    WHERE id IN ({placeholders})
                """, [access_time.isoformat()] + memory_ids)
                conn.commit()
        except Exception as e:
            logger.error(f"Error batch updating memory access times: {e}")
    
    def get_importance_since(self, since: datetime = None, exclude_types: list[str] = None) -> int:
        """Get sum of importance scores for memories created since a given time."""
        if since is None:
            since = datetime.now() - timedelta(hours=24)
        
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                if exclude_types:
                    placeholders = ",".join("?" for _ in exclude_types)
                    cursor.execute(f"""
                        SELECT COALESCE(SUM(importance_score), 0)
                        FROM memories 
                        WHERE agent_name = ? AND creation_timestamp > ?
                        AND memory_type NOT IN ({placeholders})
                    """, (self.agent_name, since.isoformat(), *exclude_types))
                else:
                    cursor.execute("""
                        SELECT COALESCE(SUM(importance_score), 0)
                        FROM memories 
                        WHERE agent_name = ? AND creation_timestamp > ?
                    """, (self.agent_name, since.isoformat()))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error calculating importance since {since}: {e}")
            return 0

    def search_memories_fts(self, query: str, limit: int = 20) -> list[Memory]:
        """Search memories using FTS5 full-text search."""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT m.* FROM memories m
                    JOIN memories_fts fts ON CAST(fts.memory_id AS INTEGER) = m.id
                    WHERE m.agent_name = ? AND memories_fts MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                """, (self.agent_name, query, limit))
                rows = cursor.fetchall()
                
                return [self._row_to_memory(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
