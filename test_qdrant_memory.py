#!/usr/bin/env python3
"""Standalone test comparing TF-IDF vs Qdrant semantic memory retrieval.

Run with: python test_qdrant_memory.py
"""

import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, ".")

from memory import Memory, MemoryStream
from memory_qdrant import QdrantMemoryStream


def create_sample_memories():
    """Create 10 diverse sample memories for testing."""
    base_time = datetime.now()

    memories = [
        Memory(
            id=1,
            agent_name="TestAgent",
            description="Isabella Rodriguez invited me to her Valentine's Day party at Hobbs Cafe",
            memory_type="observation",
            importance_score=8,
            creation_timestamp=base_time - timedelta(hours=2),
            location="Johnson Park"
        ),
        Memory(
            id=2,
            agent_name="TestAgent",
            description="I need to buy groceries for dinner tonight - eggs, milk, and bread",
            memory_type="plan",
            importance_score=4,
            creation_timestamp=base_time - timedelta(hours=5),
            location="Home"
        ),
        Memory(
            id=3,
            agent_name="TestAgent",
            description="Sam Moore mentioned he's working on a new research paper about economics",
            memory_type="observation",
            importance_score=6,
            creation_timestamp=base_time - timedelta(hours=1),
            location="Oak Hill College"
        ),
        Memory(
            id=4,
            agent_name="TestAgent",
            description="The weather today is sunny with a slight breeze, perfect for a walk",
            memory_type="observation",
            importance_score=2,
            creation_timestamp=base_time - timedelta(hours=3),
            location="Outside"
        ),
        Memory(
            id=5,
            agent_name="TestAgent",
            description="I had a deep conversation with Maria about life goals and dreams",
            memory_type="reflection",
            importance_score=7,
            creation_timestamp=base_time - timedelta(hours=8),
            location="The Rose and Crown Pub"
        ),
        Memory(
            id=6,
            agent_name="TestAgent",
            description="Klaus Mueller asked for help with his computer programming assignment",
            memory_type="observation",
            importance_score=5,
            creation_timestamp=base_time - timedelta(hours=4),
            location="Library"
        ),
        Memory(
            id=7,
            agent_name="TestAgent",
            description="I noticed romantic tension between Tom and Jennifer at the cafe",
            memory_type="observation",
            importance_score=6,
            creation_timestamp=base_time - timedelta(hours=6),
            location="Hobbs Cafe"
        ),
        Memory(
            id=8,
            agent_name="TestAgent",
            description="My reflection: I should spend more time on creative hobbies like painting",
            memory_type="reflection",
            importance_score=5,
            creation_timestamp=base_time - timedelta(hours=12),
            location="Home"
        ),
        Memory(
            id=9,
            agent_name="TestAgent",
            description="The pharmacy was closed when I went to pick up my prescription medicine",
            memory_type="observation",
            importance_score=4,
            creation_timestamp=base_time - timedelta(hours=7),
            location="Pharmacy"
        ),
        Memory(
            id=10,
            agent_name="TestAgent",
            description="Mayor announced a new community garden project at the town meeting",
            memory_type="observation",
            importance_score=7,
            creation_timestamp=base_time - timedelta(hours=24),
            location="Town Hall"
        ),
    ]
    return memories


def print_results(title: str, results: list, max_display: int = 5):
    """Print formatted memory retrieval results."""
    print(f"\n  {title}:")
    print("  " + "-" * 60)
    for i, (memory, score) in enumerate(results[:max_display]):
        desc = memory.description[:55] + "..." if len(memory.description) > 55 else memory.description
        print(f"  {i+1}. [{score:.3f}] {desc}")
    if not results:
        print("  (no results)")


def main():
    print("=" * 70)
    print("TF-IDF vs Qdrant Semantic Memory Retrieval Comparison")
    print("=" * 70)

    # Create sample memories
    memories = create_sample_memories()
    print(f"\nCreated {len(memories)} sample memories")

    # Initialize both memory systems
    print("\nInitializing TF-IDF memory stream...")
    import tempfile, os
    tmp_db = tempfile.mktemp(suffix=".db")
    tfidf_stream = MemoryStream(agent_name="TestAgent", db_path=tmp_db)
    tfidf_stream._init_database()  # create schema

    print("Initializing Qdrant memory stream...")
    qdrant_stream = QdrantMemoryStream(agent_name="TestAgent")

    # Add memories to both systems
    print("Adding memories to both systems...")
    for memory in memories:
        # For TF-IDF, we need to use SQLite - create in-memory DB
        tfidf_stream.add_memory(memory)
        qdrant_stream.add_memory(memory)

    # Reload TF-IDF vectors
    tfidf_stream._load_memories_for_vectorization()

    print(f"\nTF-IDF memory count: {len(tfidf_stream.get_memories())}")
    print(f"Qdrant memory count: {qdrant_stream.count()}")

    # Define test queries
    queries = [
        "Valentine's Day party celebration",
        "academic research and studying",
        "romantic relationships and love",
    ]

    current_time = datetime.now()

    # Run comparisons
    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: \"{query}\"")
        print("=" * 70)

        # TF-IDF retrieval
        tfidf_results = tfidf_stream.retrieve_memories(query, top_k=5, current_time=current_time)
        print_results("TF-IDF Results", tfidf_results)

        # Qdrant retrieval
        qdrant_results = qdrant_stream.retrieve_memories(query, top_k=5, current_time=current_time)
        print_results("Qdrant Semantic Results", qdrant_results)

        # Compare top results
        if tfidf_results and qdrant_results:
            tfidf_top = tfidf_results[0][0].description[:40]
            qdrant_top = qdrant_results[0][0].description[:40]
            match = "SAME" if tfidf_results[0][0].id == qdrant_results[0][0].id else "DIFFERENT"
            print(f"\n  Top result comparison: {match}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
