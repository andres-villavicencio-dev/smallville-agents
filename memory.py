"""Memory stream implementation for generative agents."""
import sqlite3
import json
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import MEMORY_RETRIEVAL_WEIGHTS, RECENCY_DECAY_FACTOR

logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """A single memory object."""
    id: Optional[int] = None
    agent_name: str = ""
    description: str = ""
    memory_type: str = "observation"  # observation, reflection, plan
    importance_score: int = 5
    creation_timestamp: datetime = None
    last_access_timestamp: datetime = None
    location: str = ""
    source_memory_ids: List[int] = None  # For reflections
    
    def __post_init__(self):
        if self.creation_timestamp is None:
            self.creation_timestamp = datetime.now()
        if self.last_access_timestamp is None:
            self.last_access_timestamp = self.creation_timestamp
        if self.source_memory_ids is None:
            self.source_memory_ids = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['creation_timestamp'] = self.creation_timestamp.isoformat()
        data['last_access_timestamp'] = self.last_access_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
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
        self._init_database()
        self._load_memories_for_vectorization()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                
                # Create FTS5 table for text search (standalone, not content-sync)
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        memory_id UNINDEXED,
                        description
                    )
                """)
                
                # Create triggers to keep FTS in sync
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(memory_id, description) VALUES (new.id, new.description);
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, memory_id, description) VALUES ('delete', old.id, old.description);
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF description ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, memory_id, description) VALUES ('delete', old.id, old.description);
                        INSERT INTO memories_fts(memory_id, description) VALUES (new.id, new.description);
                    END
                """)
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def add_memory(self, memory: Memory) -> int:
        """Add a memory to the stream."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                
                # Update vectorization
                self._load_memories_for_vectorization()
                
                return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return -1
    
    def get_memories(self, limit: Optional[int] = None, 
                    memory_type: Optional[str] = None) -> List[Memory]:
        """Get memories for the agent."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                
                memories = []
                for row in rows:
                    memory = Memory(
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
                    memories.append(memory)
                
                return memories
        except Exception as e:
            logger.error(f"Error getting memories: {e}")
            return []
    
    def _load_memories_for_vectorization(self):
        """Load all memories for TF-IDF vectorization."""
        memories = self.get_memories()
        if not memories:
            return
        
        self.memory_descriptions = [m.description for m in memories]
        try:
            self.memory_vectors = self.vectorizer.fit_transform(self.memory_descriptions)
        except Exception as e:
            logger.warning(f"Error creating memory vectors: {e}")
            self.memory_vectors = None
    
    def _calculate_recency_score(self, memory: Memory, current_time: datetime) -> float:
        """Calculate recency score using exponential decay."""
        hours_since_access = (current_time - memory.last_access_timestamp).total_seconds() / 3600
        return math.pow(RECENCY_DECAY_FACTOR, hours_since_access)
    
    def _calculate_importance_score(self, memory: Memory) -> float:
        """Normalize importance score to [0, 1]."""
        return (memory.importance_score - 1) / 9.0  # 1-10 scale to 0-1
    
    def _calculate_relevance_score(self, memory: Memory, query: str) -> float:
        """Calculate relevance score using cosine similarity."""
        if self.memory_vectors is None or not query.strip():
            return 0.5  # Default relevance
        
        try:
            query_vector = self.vectorizer.transform([query])
            memories = self.get_memories()
            
            # Find the memory index
            memory_idx = None
            for i, mem in enumerate(memories):
                if mem.id == memory.id:
                    memory_idx = i
                    break
            
            if memory_idx is not None and memory_idx < self.memory_vectors.shape[0]:
                similarity = cosine_similarity(query_vector, self.memory_vectors[memory_idx:memory_idx+1])
                return float(similarity[0][0])
        except Exception as e:
            logger.warning(f"Error calculating relevance: {e}")
        
        return 0.5
    
    def retrieve_memories(self, query: str, top_k: int = 10, 
                         current_time: Optional[datetime] = None) -> List[Tuple[Memory, float]]:
        """Retrieve top-k most relevant memories with combined scoring."""
        if current_time is None:
            current_time = datetime.now()
        
        memories = self.get_memories()
        if not memories:
            return []
        
        # Update access timestamps for retrieved memories
        memory_scores = []
        
        for memory in memories:
            # Update last access time
            memory.last_access_timestamp = current_time
            self._update_memory_access_time(memory.id, current_time)
            
            # Calculate combined score
            recency = self._calculate_recency_score(memory, current_time)
            importance = self._calculate_importance_score(memory)
            relevance = self._calculate_relevance_score(memory, query)
            
            # Normalize scores to [0, 1] and combine
            weights = MEMORY_RETRIEVAL_WEIGHTS
            combined_score = (
                weights["recency"] * recency +
                weights["importance"] * importance +
                weights["relevance"] * relevance
            ) / sum(weights.values())
            
            memory_scores.append((memory, combined_score))
        
        # Sort by combined score and return top-k
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        return memory_scores[:top_k]
    
    def _update_memory_access_time(self, memory_id: int, access_time: datetime):
        """Update the last access time for a memory."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE memories 
                    SET last_access_timestamp = ?
                    WHERE id = ?
                """, (access_time.isoformat(), memory_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating memory access time: {e}")
    
    def get_importance_since(self, since: datetime = None, exclude_types: List[str] = None) -> int:
        """Get sum of importance scores for memories created since a given time."""
        if since is None:
            since = datetime.now() - timedelta(hours=24)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
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

    def get_recent_importance_sum(self, hours: int = 24, exclude_types: List[str] = None) -> int:
        """Get sum of importance scores for recent memories."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if exclude_types:
                    placeholders = ",".join("?" for _ in exclude_types)
                    cursor.execute(f"""
                        SELECT SUM(importance_score) 
                        FROM memories 
                        WHERE agent_name = ? AND creation_timestamp > ?
                        AND memory_type NOT IN ({placeholders})
                    """, (self.agent_name, cutoff_time.isoformat(), *exclude_types))
                else:
                    cursor.execute("""
                        SELECT SUM(importance_score) 
                        FROM memories 
                        WHERE agent_name = ? AND creation_timestamp > ?
                    """, (self.agent_name, cutoff_time.isoformat()))
                result = cursor.fetchone()
                return result[0] if result[0] is not None else 0
        except Exception as e:
            logger.error(f"Error calculating recent importance sum: {e}")
            return 0
    
    def search_memories_fts(self, query: str, limit: int = 20) -> List[Memory]:
        """Search memories using FTS5 full-text search."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT m.* FROM memories m
                    JOIN memories_fts fts ON CAST(fts.memory_id AS INTEGER) = m.id
                    WHERE m.agent_name = ? AND memories_fts MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                """, (self.agent_name, query, limit))
                rows = cursor.fetchall()
                
                memories = []
                for row in rows:
                    memory = Memory(
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
                    memories.append(memory)
                
                return memories
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []