"""Qdrant-based semantic memory stream for generative agents.

Uses sentence-transformers for embeddings and Qdrant for vector search,
with recency/importance re-scoring to match the original paper's retrieval formula.
"""

import sqlite3
import json
import math
import logging
from datetime import datetime
from typing import List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from config import MEMORY_RETRIEVAL_WEIGHTS, RECENCY_DECAY_FACTOR
from memory import Memory

logger = logging.getLogger(__name__)

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


class QdrantMemoryStream:
    """Qdrant-backed memory stream with semantic search and paper-based re-scoring."""

    def __init__(self, agent_name: str, collection_name: str = None):
        """Initialize Qdrant memory stream.

        Args:
            agent_name: Name of the agent this memory stream belongs to.
            collection_name: Qdrant collection name. Defaults to agent_name.
        """
        self.agent_name = agent_name
        self.collection_name = collection_name or f"memories_{agent_name.lower().replace(' ', '_')}"

        # In-memory Qdrant for fast prototyping (no persistence)
        self.client = QdrantClient(":memory:")

        # Sentence transformer for embeddings
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Memory metadata cache (id -> Memory object)
        self._memory_cache: dict[int, Memory] = {}

        self._init_collection()

    def _init_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def add_memory(self, memory: Memory) -> int:
        """Add a memory to the Qdrant index.

        Args:
            memory: Memory object to add.

        Returns:
            The memory ID (uses memory.id if set, otherwise generates one).
        """
        # Generate ID if not provided
        if memory.id is None:
            # Use timestamp-based ID for uniqueness
            memory.id = int(datetime.now().timestamp() * 1000000) % (2**63)

        # Embed the description
        embedding = self.encoder.encode(memory.description).tolist()

        # Store in Qdrant with metadata
        point = PointStruct(
            id=memory.id,
            vector=embedding,
            payload={
                "agent_name": memory.agent_name,
                "description": memory.description,
                "memory_type": memory.memory_type,
                "importance_score": memory.importance_score,
                "creation_timestamp": memory.creation_timestamp.isoformat(),
                "last_access_timestamp": memory.last_access_timestamp.isoformat(),
                "location": memory.location,
                "source_memory_ids": memory.source_memory_ids or []
            }
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        # Cache the memory object
        self._memory_cache[memory.id] = memory

        logger.debug(f"Added memory {memory.id}: {memory.description[:50]}...")
        return memory.id

    def _payload_to_memory(self, point_id: int, payload: dict) -> Memory:
        """Convert Qdrant payload to Memory object."""
        return Memory(
            id=point_id,
            agent_name=payload["agent_name"],
            description=payload["description"],
            memory_type=payload["memory_type"],
            importance_score=payload["importance_score"],
            creation_timestamp=datetime.fromisoformat(payload["creation_timestamp"]),
            last_access_timestamp=datetime.fromisoformat(payload["last_access_timestamp"]),
            location=payload.get("location", ""),
            source_memory_ids=payload.get("source_memory_ids", [])
        )

    def _calculate_recency_score(self, memory: Memory, current_time: datetime) -> float:
        """Calculate recency score using exponential decay (matches MemoryStream)."""
        hours_since_access = (current_time - memory.last_access_timestamp).total_seconds() / 3600
        return math.pow(RECENCY_DECAY_FACTOR, hours_since_access)

    def _calculate_importance_score(self, memory: Memory) -> float:
        """Normalize importance score to [0, 1] (matches MemoryStream)."""
        return (memory.importance_score - 1) / 9.0  # 1-10 scale to 0-1

    def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        current_time: Optional[datetime] = None
    ) -> List[Tuple[Memory, float]]:
        """Retrieve top-k most relevant memories using semantic search + re-scoring.

        Uses Qdrant for initial semantic retrieval, then re-scores results using
        the paper's formula: alpha*recency + beta*importance + gamma*relevance

        Args:
            query: The query text to search for.
            top_k: Number of memories to return.
            current_time: Current simulation time (defaults to now).

        Returns:
            List of (Memory, score) tuples sorted by combined score descending.
        """
        if current_time is None:
            current_time = datetime.now()

        # Get collection info to check if empty
        collection_info = self.client.get_collection(self.collection_name)
        if collection_info.points_count == 0:
            return []

        # Embed the query
        query_embedding = self.encoder.encode(query).tolist()

        # Retrieve more candidates than needed for re-scoring (3x top_k or at least 20)
        search_limit = max(top_k * 3, 20)

        # Semantic search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=search_limit
        )

        if not search_results:
            return []

        # Re-score using the paper's retrieval formula
        memory_scores = []
        weights = MEMORY_RETRIEVAL_WEIGHTS
        weight_sum = sum(weights.values())

        for result in search_results:
            memory = self._payload_to_memory(result.id, result.payload)

            # Semantic similarity from Qdrant (already cosine similarity)
            relevance = result.score

            # Recency and importance scores
            recency = self._calculate_recency_score(memory, current_time)
            importance = self._calculate_importance_score(memory)

            # Combined score (same formula as MemoryStream)
            combined_score = (
                weights["recency"] * recency +
                weights["importance"] * importance +
                weights["relevance"] * relevance
            ) / weight_sum

            memory_scores.append((memory, combined_score))

        # Sort by combined score and take top_k
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        return memory_scores[:top_k]

    def sync_from_sqlite(self, db_path: str = "db/memories.db") -> int:
        """Rebuild Qdrant index from SQLite database.

        Args:
            db_path: Path to the SQLite memories database.

        Returns:
            Number of memories indexed.
        """
        logger.info(f"Syncing memories from {db_path} for agent {self.agent_name}")

        # Clear existing collection and recreate
        self.client.delete_collection(self.collection_name)
        self._init_collection()
        self._memory_cache.clear()

        try:
            with sqlite3.connect(db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, agent_name, description, memory_type, importance_score,
                           creation_timestamp, last_access_timestamp, location, source_memory_ids
                    FROM memories
                    WHERE agent_name = ?
                    ORDER BY creation_timestamp ASC
                """, (self.agent_name,))
                rows = cursor.fetchall()

                if not rows:
                    logger.info(f"No memories found for agent {self.agent_name}")
                    return 0

                # Batch embed all descriptions
                descriptions = [row[2] for row in rows]
                embeddings = self.encoder.encode(descriptions)

                # Prepare points for batch upsert
                points = []
                for i, row in enumerate(rows):
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

                    points.append(PointStruct(
                        id=memory.id,
                        vector=embeddings[i].tolist(),
                        payload={
                            "agent_name": memory.agent_name,
                            "description": memory.description,
                            "memory_type": memory.memory_type,
                            "importance_score": memory.importance_score,
                            "creation_timestamp": memory.creation_timestamp.isoformat(),
                            "last_access_timestamp": memory.last_access_timestamp.isoformat(),
                            "location": memory.location,
                            "source_memory_ids": memory.source_memory_ids
                        }
                    ))

                    self._memory_cache[memory.id] = memory

                # Batch upsert
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

                logger.info(f"Indexed {len(points)} memories for agent {self.agent_name}")
                return len(points)

        except Exception as e:
            logger.error(f"Error syncing from SQLite: {e}")
            raise

    def get_all_memories(self) -> List[Memory]:
        """Get all memories in the collection."""
        collection_info = self.client.get_collection(self.collection_name)
        if collection_info.points_count == 0:
            return []

        # Scroll through all points
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        return [self._payload_to_memory(r.id, r.payload) for r in results]

    def count(self) -> int:
        """Return the number of memories in the collection."""
        return self.client.get_collection(self.collection_name).points_count
