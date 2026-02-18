# Paper Result Reproduction - Generative Agents: Interactive Simulacra of Human Behavior

A faithful implementation of the Stanford paper "Generative Agents: Interactive Simulacra of Human Behavior" (arXiv:2304.03442) using local Ollama models instead of OpenAI.

## Overview

This project recreates the groundbreaking generative agents architecture that enables believable human-like behavior in AI agents through:

- **Memory Stream**: Each agent maintains episodic memories with retrieval based on recency, importance, and relevance
- **Reflection**: Agents generate high-level insights from their experiences 
- **Planning**: Hierarchical daily plans that decompose into specific actions
- **Social Interactions**: Natural conversations between agents based on memory and context

## Architecture

### Core Components

1. **Memory System** (`memory.py`)
   - SQLite storage with FTS5 full-text search
   - TF-IDF vectorization for semantic similarity
   - Weighted retrieval combining recency, importance, and relevance scores
   - Exponential decay for recency, LLM scoring for importance

2. **Agent Class** (`agent.py`)
   - Individual agents with unique personas and backgrounds
   - Daily planning with hierarchical decomposition
   - Reflection triggered by importance threshold
   - Memory-informed decision making

3. **Environment** (`environment.py`)  
   - Smallville world with 12 locations and sub-areas
   - Agent movement and location tracking
   - Environmental observations and object interactions

4. **Conversation System** (`conversation.py`)
   - Memory-driven dialogue between agents
   - Conversation initiation based on relevance and context
   - Turn-taking with personality-informed responses

5. **LLM Interface** (`llm.py`)
   - Local Ollama integration with GPU queue support
   - Pi 5 fallback for when main GPU is busy
   - Templated prompts faithful to the original paper

### The Smallville World

**Locations:**
- **Residential**: Lin Family Home, Moreno Family Home, Moore Family Home, The Willows (apartments)
- **Commercial**: Harvey Oak Supply Store, Hobbs Cafe, The Rose and Crown Pub, Pharmacy  
- **Public**: Oak Hill College, Johnson Park, Town Hall, Library

**25 Unique Agents** including:
- **John Lin** - Pharmacy owner, family man
- **Isabella Rodriguez** - Cafe owner planning Valentine's Day party  
- **Eddy Lin** - Music student, son of John and Mei
- **Sam Moore** - Mayoral candidate
- **Dr. Williams** - Town doctor and mentor
- ...and 20 more with rich backstories and relationships

## Installation & Setup

### Prerequisites

1. **Ollama** - Install and start Ollama with a compatible model
```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull recommended model
ollama pull qwen2.5:3b
```

2. **Python 3.11+**
```bash
pip install -r requirements.txt
```

### Dependencies
```
requests>=2.31.0
rich>=13.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## Running the Simulation

### Quick Start
```bash
# Run with defaults (qwen2.5:3b, 10x speed, 2 days)
python main.py

# Custom configuration
python main.py --speed 5 --days 3

# Use different model
OLLAMA_MODEL=llama3.2:3b python main.py

# Run without GPU queue (direct Ollama calls)
python main.py --no-gpu-queue
```

### Configuration Options

**Environment Variables:**
- `OLLAMA_MODEL` - Model to use (default: qwen2.5:3b)
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)

**Command Line Arguments:**
- `--speed N` - Simulation speed (game seconds per real second)
- `--days N` - Number of simulated days to run
- `--no-gpu-queue` - Disable GPU queue integration
- `--load-state FILE` - Resume from saved state
- `--config` - Show current configuration

### The Valentine's Day Scenario

The simulation starts on February 13th with Isabella Rodriguez having a memory that she's planning a Valentine's Day party at Hobbs Cafe on February 14th, 5-7 PM. Watch as:

- Isabella invites people to the party
- Word spreads through natural conversations
- Agents make plans to attend
- Social dynamics emerge organically

## GPU Queue Integration

The system integrates with OpenClaw's GPU queue for efficient resource management:

```python
# Automatic GPU queueing
response = await llm.generate("What are your thoughts on today?")

# Manual queueing for intensive operations
with gpu_queue("reflection_task"):
    reflection = await agent.reflect()
```

**Pi 5 Fallback:** If the main GPU is busy, queries automatically fall back to a Raspberry Pi 5 running gemma3:1b for continued operation.

## Memory System Details

### Memory Types
- **Observations**: Perceptions of environment and other agents
- **Reflections**: High-level insights generated from observations  
- **Plans**: Intended actions and schedules

### Retrieval Scoring
Each memory gets a combined score from:
- **Recency**: Exponential decay based on last access (α = 1.0)
- **Importance**: 1-10 scale rated by LLM (β = 1.0)  
- **Relevance**: Cosine similarity to query (γ = 1.0)

**Final Score** = α×recency + β×importance + γ×relevance

### Reflection Process
1. Sum recent observation importance scores
2. When threshold exceeded (150), trigger reflection
3. Generate 3 salient questions from last 100 memories
4. For each question, retrieve relevant memories
5. Synthesize reflection citing specific memories
6. Store reflection as high-importance memory

## Planning System

### Hierarchical Planning
1. **Daily Overview**: Generate 5-8 broad activity chunks
2. **Action Decomposition**: Break chunks into 5-15 minute actions
3. **Execution**: Follow plan while adapting to conversations and observations
4. **Replanning**: Revise plans based on new information

### Example Daily Plan
```
6:00 AM - Wake up and morning routine (30 min)
6:30 AM - Breakfast with family (45 min)  
7:15 AM - Commute to work (15 min)
7:30 AM - Open pharmacy and prep for day (60 min)
...
```

## Conversation System

### Initiation Conditions
- Agents in same location
- Memory retrieval suggests relevance
- LLM determines appropriateness
- Consider relationship and context

### Dialogue Generation
- Retrieve relevant memories about the other agent
- Consider personality and current emotional state
- Generate contextual responses
- Store conversation as memories for both participants

## Display & Monitoring

Rich terminal interface showing:
- **Real-time Agent Activities**: What each agent is doing and where
- **Location Populations**: Who is at each location  
- **Active Conversations**: Ongoing dialogues between agents
- **Recent Events**: Activity log with timestamps
- **Simulation Time**: Current game time and simulation speed

### Logs and State
- **simulation.log**: Detailed logging of all agent activities
- **saves/**: Automatic state saves for resuming simulation
- **db/memories.db**: SQLite database of all agent memories

## Research Fidelity

This implementation closely follows the original paper:

### Memory Stream Architecture ✓
- Weighted retrieval (recency + importance + relevance)
- Exponential decay functions
- FTS5 for efficient text search
- Memory access timestamp updates

### Reflection Implementation ✓  
- Importance threshold triggering (150 points)
- Question generation from recent memories
- Memory citing in reflections
- High importance scoring for reflections

### Planning System ✓
- Hierarchical decomposition (daily → actions)
- Time-based scheduling
- Location-aware planning
- Plan revision based on observations

### Agent Architecture ✓
- 25 unique personas with relationships
- Personality-driven behavior
- Memory-informed decision making
- Social interaction capabilities

## Customization

### Adding New Agents
Edit `personas.py` to add new characters:
```python
"New Agent": {
    "name": "New Agent",
    "age": 30,
    "occupation": "Job Title",
    "personality": "Trait list",
    "background": "Background story",
    "relationships": {"Other Agent": "relationship_type"},
    "home_location": "Location Name",
    "work_location": "Location Name"
}
```

### Modifying Locations  
Edit `config.py` to add new locations:
```python
SMALLVILLE_LOCATIONS["New Location"] = ["sub_area_1", "sub_area_2"]
```

### Adjusting Parameters
Key parameters in `config.py`:
- `MEMORY_RETRIEVAL_WEIGHTS`: Adjust α, β, γ for memory retrieval
- `IMPORTANCE_THRESHOLD`: Change reflection trigger (default: 150)
- `RECENCY_DECAY_FACTOR`: Modify memory decay rate (default: 0.99)

## Performance & Scaling

**Recommended Hardware:**
- 8GB+ RAM for full 25-agent simulation  
- GPU with 6GB+ VRAM for qwen2.5:3b
- SSD for fast SQLite database operations

**Scaling Options:**
- Reduce agent count for faster simulation
- Use smaller models (llama3.2:1b) for resource-constrained environments
- Adjust tick duration and simulation speed

## Troubleshooting

### Common Issues

**Ollama Connection Errors:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve
```

**Memory Database Locked:**
- Stop all simulation instances
- Check for zombie processes: `ps aux | grep python`
- Delete lock files if present: `rm db/*.db-*`

**GPU Queue Issues:**
- Check queue status: `python queue_manager.py status`
- Run without GPU queue: `python main.py --no-gpu-queue`

### Debug Mode
Enable detailed logging:
```python
# In main.py
logging.getLogger().setLevel(logging.DEBUG)
```

## Contributing

This implementation aims for research reproducibility. When contributing:

1. Maintain fidelity to the original paper methodology
2. Document any deviations or optimizations  
3. Include tests for new functionality
4. Update docstrings and type hints

## License

MIT License - See LICENSE file for details

## Citation

If you use this implementation in research, please cite both:

```bibtex
@article{park2023generative,
  title={Generative Agents: Interactive Simulacra of Human Behavior},
  author={Park, Joon Sung and O'Brien, Joseph C and Cai, Carrie Jun and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S},
  journal={arXiv preprint arXiv:2304.03442},
  year={2023}
}
```

## Acknowledgments

- Original Generative Agents research by Stanford HCI and Google Research
- OpenClaw framework for LLM integration and GPU queue management
- Ollama project for local LLM inference
- Rich library for beautiful terminal interfaces

---

**🏠 Welcome to Smallville - Where AI agents live, learn, and interact!**
