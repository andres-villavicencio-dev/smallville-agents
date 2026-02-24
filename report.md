# Generative Agents — Codebase Optimization Report

**Generated:** 2026-02-24
**Codebase:** ~7,000 lines Python across 12+ core modules
**Scope:** Performance, data layer, reliability, security, testing, code quality

---

## 1. Executive Summary

This report documents **31 optimization opportunities** identified across the Generative Agents simulation codebase. Findings are categorized by severity and domain:

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 3 | Blocks scalability or exposes security risk |
| HIGH | 9 | Significant performance or reliability impact |
| MEDIUM | 10 | Measurable improvement, moderate effort |
| LOW | 9 | Minor polish or future-proofing |

**Top 7 Quick Wins** (under 1 hour each):

1. **PERF-01**: Wrap `ollama_query()` in `run_in_executor()` — unblocks entire event loop (1h)
2. **DATA-03**: Add `PRAGMA journal_mode=WAL` — eliminates write stalls (30m)
3. **SEC-02**: Bind WebUI to `127.0.0.1` — closes network exposure (15m)
4. **DATA-06**: Add `timeout=30` to `sqlite3.connect()` — prevents lock errors (15m)
5. **CQ-03**: Add `RotatingFileHandler` — prevents disk fill (15m)
6. **PERF-04**: `asyncio.gather()` for planning_thought + memo_thought — 2x faster post-conversation (30m)
7. **CQ-02**: Fix token budget inflation — saves compute on short tasks (15m)

**Estimated total effort:** ~40-50 developer-hours

---

## 2. Priority Matrix

| ID | Description | File:Line | Severity | Effort | Category |
|----|-------------|-----------|----------|--------|----------|
| PERF-01 | GPU queue `ollama_query()` blocks asyncio event loop | `llm.py:78` | CRITICAL | 1h | Performance |
| PERF-02 | New `aiohttp.ClientSession` per LLM call | `llm.py:108`, `committee.py:279` | HIGH | 2h | Performance |
| PERF-03 | Conversations advance sequentially (for-loop, not gather) | `conversation.py:343-346` | HIGH | 2h | Performance |
| PERF-04 | 3 sequential LLM calls in `react_to_conversation()` | `agent.py:631-648` | MEDIUM | 30m | Performance |
| PERF-05 | `LLM_SEMAPHORE_LIMIT` defined but never used | `config.py:57` | LOW | 30m | Performance |
| PERF-06 | Stats recount fetches all memories for 25 agents every 10 ticks | `main.py:574-580` | MEDIUM | 1h | Performance |
| PERF-07 | O(n²) pair generation before sampling | `main.py:399-403` | LOW | 30m | Performance |
| DATA-01 | TF-IDF full vectorizer rebuild on every retrieval after `add_memory()` | `memory.py:165-166` | CRITICAL | 4h | Data Layer |
| DATA-02 | Skill vector full rebuild on every `add_skill()` | `skillbank.py:123` | HIGH | 2h | Data Layer |
| DATA-03 | No SQLite WAL mode — write stalls under concurrent access | `memory.py`, `skillbank.py` | HIGH | 30m | Data Layer |
| DATA-04 | No connection pooling — new `sqlite3.connect()` per operation | `memory.py`, `skillbank.py` | HIGH | 2h | Data Layer |
| DATA-05 | FTS5 join uses `CAST()` preventing index usage | `memory.py:358` | MEDIUM | 1h | Data Layer |
| DATA-06 | No connection timeout — `database is locked` under load | all `sqlite3.connect()` calls | MEDIUM | 15m | Data Layer |
| DATA-07 | Recency scoring in Python for ALL memories (no SQL pre-filter) | `memory.py:248-283` | MEDIUM | 2h | Data Layer |
| DATA-08 | `source_memory_ids` as List not Set (O(n) lookups) | `memory.py:27` | LOW | 30m | Data Layer |
| DATA-09 | `location_connections` as List not Set | `environment.py:34` | LOW | 15m | Data Layer |
| DATA-10 | Persona lookups are O(n) linear scans | `personas.py:510-513` | LOW | 30m | Data Layer |
| REL-01 | `SteeredCommittee.load_engine()` race condition on `_load_attempted` | `committee.py:338-341` | HIGH | 1h | Reliability |
| REL-02 | LLM retry loop can block event loop 14+ minutes | `llm.py:102-128`, `committee.py:273-300` | HIGH | 3h | Reliability |
| REL-03 | Non-atomic state file writes — corruption on crash | `main.py:637-643` | MEDIUM | 30m | Reliability |
| REL-04 | 10+ bare `except Exception: pass` silencing real errors | `webui.py`, `display.py`, `llm.py`, etc. | MEDIUM | 1h | Reliability |
| REL-05 | Qwen3 thinking block stripping can lose entire response | `steering/engine.py:215-226` | MEDIUM | 30m | Reliability |
| REL-06 | Telegram broadcaster uses blocking `requests.post()` | `telegram_broadcaster.py:80` | MEDIUM | N/A | Reliability |
| REL-07 | Composite direction cache grows unbounded | `steering/engine.py:38` | LOW | 30m | Reliability |
| SEC-01 | No authentication on WebUI REST API | `webui.py:82-93` | CRITICAL | 1-3h | Security |
| SEC-02 | WebUI binds to `0.0.0.0` (all interfaces) | `webui.py:46` | HIGH | 15m | Security |
| TEST-01 | Zero test coverage for 5 core modules | `llm.py`, `committee.py`, `webui.py`, `display.py`, `telegram_broadcaster.py` | HIGH | 8-12h | Testing |
| CQ-01 | Global singletons hinder testability | `llm.py:215`, `committee.py:496` | MEDIUM | 4h | Code Quality |
| CQ-02 | Token budget unconditionally inflated for Qwen models | `steering/engine.py:231-232` | LOW | 15m | Code Quality |
| CQ-03 | No log rotation — `simulation.log` grows unbounded | `main.py:25-32` | LOW | 15m | Code Quality |
| CQ-04 | Duplicate import of `IMPORTANCE_THRESHOLD` | `reflection_engine.py:12` | LOW | 1m | Code Quality |

---

## 3. Performance Bottlenecks

### PERF-01: GPU queue `ollama_query()` blocks asyncio event loop [CRITICAL]

**File:** `llm.py:78`
**Effort:** 1 hour

**Description:** The `ollama_query()` function from the GPU queue module is a synchronous blocking call. When invoked inside the `async _generate_with_queue()` method, it blocks the entire asyncio event loop, preventing all other agents from processing concurrently.

**Root cause:** `ollama_query()` is imported from `queue_manager.py` and makes synchronous HTTP requests. Calling it directly inside an `async` method stalls the event loop.

**Code (current):**
```python
# llm.py:73-82
async def _generate_with_queue(self, prompt, system_prompt, temperature, max_tokens, model):
    try:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = ollama_query(full_prompt, model=model)  # BLOCKING!
        return response.strip()
```

**Suggested fix:**
```python
async def _generate_with_queue(self, prompt, system_prompt, temperature, max_tokens, model):
    try:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: ollama_query(full_prompt, model=model)
        )
        return response.strip()
```

---

### PERF-02: New `aiohttp.ClientSession` created per LLM call [HIGH]

**Files:** `llm.py:108`, `committee.py:279`
**Effort:** 2 hours

**Description:** Every LLM call creates and destroys an `aiohttp.ClientSession`. Each session involves TCP connection setup, potential TLS negotiation, and teardown. With 25 agents making multiple calls per tick, this adds significant overhead.

**Code (current):**
```python
# llm.py:107-108
for attempt in range(max_retries):
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(...) as resp:
                ...
```

**Suggested fix:** Create a single session at `OllamaClient.__init__` and reuse it:
```python
class OllamaClient:
    def __init__(self, use_gpu_queue=True):
        self.base_url = OLLAMA_BASE_URL
        self.use_gpu_queue = use_gpu_queue and GPU_QUEUE_AVAILABLE
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
```

Apply the same pattern to `Committee._call_model()` at `committee.py:279`.

---

### PERF-03: Conversations advance sequentially [HIGH]

**File:** `conversation.py:343-346`
**Effort:** 2 hours

**Description:** In `update_conversations()`, active conversations are advanced one at a time in a for-loop. Each iteration makes an LLM call, so N active conversations take N × LLM_latency instead of max(LLM_latency).

**Code (current):**
```python
# conversation.py:343-346
for conversation in list(self.active_conversations.values()):
    if conversation.active:
        continued = await self.continue_conversation(conversation, memory_streams)
        if not continued or await self.should_end_conversation(conversation):
            conversations_to_end.append(conversation)
```

**Suggested fix:**
```python
async def _advance_one(conversation, memory_streams):
    continued = await self.continue_conversation(conversation, memory_streams)
    if not continued or await self.should_end_conversation(conversation):
        return conversation
    return None

results = await asyncio.gather(*[
    _advance_one(conv, memory_streams)
    for conv in self.active_conversations.values()
    if conv.active
], return_exceptions=True)

conversations_to_end = [r for r in results if r is not None and not isinstance(r, Exception)]
```

---

### PERF-04: 3 sequential LLM calls in `react_to_conversation()` [MEDIUM]

**File:** `agent.py:631-648`
**Effort:** 30 minutes

**Description:** After each conversation ends, the agent makes two independent LLM calls (planning thought + memo thought) sequentially, then a third for replanning. The first two are independent and can run in parallel.

**Code (current):**
```python
# agent.py:631-648
if cfg.USE_COMMITTEE:
    ...
    planning_thought = await committee._call_model(expert, planning_prompt)
    ...
    memo_thought = await committee._call_model(expert, memo_prompt)
else:
    llm = await get_llm_client()
    planning_thought = await llm.generate(planning_prompt, ...)
    memo_thought = await llm.generate(memo_prompt, ...)
```

**Suggested fix:**
```python
# Gather planning + memo thoughts in parallel
planning_thought, memo_thought = await asyncio.gather(
    llm.generate(planning_prompt, temperature=0.6, max_tokens=100, task="reflection"),
    llm.generate(memo_prompt, temperature=0.6, max_tokens=100, task="reflection"),
)
```

Note: For the committee branch with sequential VRAM constraints, parallel calls may not be possible. Apply only to the single-model branch.

---

### PERF-05: `LLM_SEMAPHORE_LIMIT` defined but never used [LOW]

**File:** `config.py:57`
**Effort:** 30 minutes

**Description:** `LLM_SEMAPHORE_LIMIT = 2` is defined in config but never referenced by any module. It was likely intended to cap concurrent Ollama requests.

**Suggested fix:** Either implement the semaphore in `llm.py` or remove the dead config entry:
```python
# In llm.py
_llm_semaphore = asyncio.Semaphore(LLM_SEMAPHORE_LIMIT)

async def _generate_direct(self, ...):
    async with _llm_semaphore:
        ...
```

---

### PERF-06: Stats recount fetches all memories for 25 agents every 10 ticks [MEDIUM]

**File:** `main.py:574-580`
**Effort:** 1 hour

**Description:** Every 10 ticks, `_update_stats()` calls `get_memories()` (no limit) for every agent twice — once for all memories and once filtered to reflections. With 25 agents and growing memory stores, this becomes a significant DB bottleneck.

**Code (current):**
```python
# main.py:574-580
if self.tick_count % 10 == 0:
    for agent in self.agents.values():
        memories = agent.memory_stream.get_memories()  # Fetches ALL
        total_memories += len(memories)
        reflections = agent.memory_stream.get_memories(memory_type="reflection")  # Fetches ALL reflections
        total_reflections += len(reflections)
```

**Suggested fix:** Use SQL `COUNT(*)` queries instead of fetching full row sets:
```python
def count_memories(self, memory_type: Optional[str] = None) -> int:
    """Count memories without loading them into Python."""
    with sqlite3.connect(self.db_path) as conn:
        if memory_type:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_name = ? AND memory_type = ?",
                (self.agent_name, memory_type)
            )
        else:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_name = ?",
                (self.agent_name,)
            )
        return cursor.fetchone()[0]
```

---

### PERF-07: O(n²) pair generation before sampling [LOW]

**File:** `main.py:399-403`
**Effort:** 30 minutes

**Description:** When checking for conversation opportunities, `_update_conversations()` generates all possible agent pairs at a location before sampling down to `MAX_CONV_CHECKS_PER_LOCATION`. With 10 agents at one location, this creates 45 pairs only to sample 3.

**Code (current):**
```python
# main.py:399-403
all_pairs = [
    (agents_here[i], agents_here[j])
    for i in range(len(agents_here))
    for j in range(i+1, len(agents_here))
]
max_checks = getattr(cfg, 'MAX_CONV_CHECKS_PER_LOCATION', 3)
if len(all_pairs) > max_checks:
    all_pairs = random.sample(all_pairs, max_checks)
```

**Suggested fix:** Sample indices directly instead of generating all pairs:
```python
import random
n = len(agents_here)
max_checks = getattr(cfg, 'MAX_CONV_CHECKS_PER_LOCATION', 3)
if n * (n - 1) // 2 <= max_checks:
    pairs = [(agents_here[i], agents_here[j]) for i in range(n) for j in range(i+1, n)]
else:
    pairs = set()
    while len(pairs) < max_checks:
        i, j = sorted(random.sample(range(n), 2))
        pairs.add((agents_here[i], agents_here[j]))
    pairs = list(pairs)
```

---

## 4. Data Layer & Memory System

### DATA-01: TF-IDF full vectorizer rebuild on every retrieval after `add_memory()` [CRITICAL]

**File:** `memory.py:165-166`
**Effort:** 4 hours

**Description:** Every `add_memory()` call sets `_vector_cache_valid = False`. On the next `retrieve_memories()` call, `_load_memories_for_vectorization()` is triggered, which loads ALL memories from the database, rebuilds the TF-IDF vectorizer from scratch via `fit_transform()`, and recomputes all vectors. With 25 agents generating observations each tick, this happens hundreds of times per simulation day.

**Code (current):**
```python
# memory.py:165-166
# Invalidate vector cache (will rebuild on next retrieval)
self._vector_cache_valid = False
```

Then in `retrieve_memories()`:
```python
# memory.py:253-254
if not self._vector_cache_valid:
    self._load_memories_for_vectorization()  # Full rebuild!
```

**Suggested fix:** Use incremental updates instead of full rebuilds:
```python
def add_memory(self, memory: Memory) -> int:
    ...
    memory_id = cursor.lastrowid
    conn.commit()

    # Incremental: append to existing vectors instead of full rebuild
    self.memory_descriptions.append(memory.description)
    if self.memory_vectors is not None and len(self.memory_descriptions) > 1:
        new_vec = self.vectorizer.transform([memory.description])
        from scipy.sparse import vstack
        self.memory_vectors = vstack([self.memory_vectors, new_vec])
    else:
        self._vector_cache_valid = False  # Full rebuild only when vectorizer not fitted
    return memory_id
```

Note: `transform()` (not `fit_transform()`) uses the existing vocabulary, so new documents can be added incrementally. A periodic full rebuild (e.g., every 100 additions) ensures vocabulary stays fresh.

---

### DATA-02: Skill vector full rebuild on every `add_skill()` [HIGH]

**File:** `skillbank.py:123`
**Effort:** 2 hours

**Description:** Same pattern as DATA-01. Every `add_skill()` calls `self._rebuild_vectors()` which fetches all skills and calls `fit_transform()` from scratch.

**Code (current):**
```python
# skillbank.py:122-123
skill_id = cursor.lastrowid
self._rebuild_vectors()  # Full rebuild!
```

**Suggested fix:** Apply the same incremental `transform()` + `vstack()` approach as DATA-01.

---

### DATA-03: No SQLite WAL mode [HIGH]

**Files:** `memory.py`, `skillbank.py`
**Effort:** 30 minutes

**Description:** SQLite defaults to journal mode `DELETE`, which takes an exclusive lock during writes. With 25 agents writing memories concurrently, this causes "database is locked" errors and write stalls. WAL (Write-Ahead Logging) mode allows concurrent reads during writes.

**Suggested fix:** Add WAL pragma after every connection open:
```python
def _init_database(self):
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # Safe with WAL, faster than FULL
        ...
```

---

### DATA-04: No connection pooling [HIGH]

**Files:** `memory.py`, `skillbank.py`
**Effort:** 2 hours

**Description:** Every database operation creates a new `sqlite3.connect()` call. While SQLite connections are lightweight, the overhead of opening/closing connections hundreds of times per tick adds up, especially with WAL mode where persistent connections perform better.

**Suggested fix:** Maintain a persistent connection per `MemoryStream`/`SkillBank` instance:
```python
class MemoryStream:
    def __init__(self, agent_name, db_path="db/memories.db"):
        self.agent_name = agent_name
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_database()
        ...

    def _get_conn(self):
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
```

---

### DATA-05: FTS5 join uses `CAST()` preventing index usage [MEDIUM]

**File:** `memory.py:358`
**Effort:** 1 hour

**Description:** The FTS5 search join uses `CAST(fts.memory_id AS INTEGER) = m.id`, which prevents SQLite from using the index on `memories.id` for the join. The `CAST()` forces a full scan of the FTS results.

**Code (current):**
```python
# memory.py:356-362
cursor.execute("""
    SELECT m.* FROM memories m
    JOIN memories_fts fts ON CAST(fts.memory_id AS INTEGER) = m.id
    WHERE m.agent_name = ? AND memories_fts MATCH ?
    ORDER BY fts.rank
    LIMIT ?
""", (self.agent_name, query, limit))
```

**Suggested fix:** Store `memory_id` as INTEGER in the FTS5 table (FTS5 UNINDEXED columns preserve type), then remove the CAST:
```sql
JOIN memories_fts fts ON fts.memory_id = m.id
```

If the column was originally inserted as TEXT, migrate existing data or use a content-sync FTS5 table tied to the main table's rowid.

---

### DATA-06: No connection timeout [MEDIUM]

**Files:** All `sqlite3.connect()` calls
**Effort:** 15 minutes

**Description:** All `sqlite3.connect()` calls use the default 5-second timeout. Under high concurrency (25 agents writing simultaneously), this is often too short, causing `OperationalError: database is locked`.

**Suggested fix:** Add a 30-second timeout globally:
```python
sqlite3.connect(self.db_path, timeout=30)
```

---

### DATA-07: Recency scoring in Python for ALL memories [MEDIUM]

**File:** `memory.py:248-283`
**Effort:** 2 hours

**Description:** `retrieve_memories()` loads ALL memories from the database, then computes recency scores in Python. As memory stores grow (thousands of memories per agent), this becomes a bottleneck.

**Code (current):**
```python
# memory.py:248-249
memories = self.get_memories()  # Loads ALL memories
if not memories:
    return []
```

**Suggested fix:** Pre-filter to recent memories in SQL before scoring in Python:
```python
# Only load memories from the last N hours for scoring
memories = self.get_memories(limit=200)  # Cap at 200 most recent
```

Or push recency filtering into SQL with a WHERE clause on `last_access_timestamp`.

---

### DATA-08: `source_memory_ids` as List not Set [LOW]

**File:** `memory.py:27`
**Effort:** 30 minutes

**Description:** `source_memory_ids` is typed as `List[int]` but is used for membership checks (reflection source tracking). Lists have O(n) lookup vs O(1) for sets.

**Impact:** Minimal for current usage (reflections rarely have >10 source memories), but would matter at scale.

---

### DATA-09: `location_connections` as List not Set [LOW]

**File:** `environment.py:34`
**Effort:** 15 minutes

**Description:** `location_connections` maps each location to a `List[str]` of connected locations. Since Smallville is fully connected (all 12 locations link to all 11 others), this is mostly unused, but if neighbor checks were added, sets would give O(1) lookups.

---

### DATA-10: Persona lookups are O(n) linear scans [LOW]

**File:** `personas.py:510-513`
**Effort:** 30 minutes

**Description:** `get_agents_by_location()` does a linear scan of all 25 personas to find agents at a given location. With only 25 agents this is negligible, but the pattern wouldn't scale.

**Code (current):**
```python
# personas.py:510-513
for name, persona in AGENT_PERSONAS.items():
    if (persona.get("home_location") == location or
        persona.get("work_location") == location):
        agents.append(name)
```

**Suggested fix:** Build a reverse index at module load time:
```python
_LOCATION_INDEX: Dict[str, List[str]] = {}

def _build_location_index():
    for name, persona in AGENT_PERSONAS.items():
        for key in ("home_location", "work_location"):
            loc = persona.get(key)
            if loc:
                _LOCATION_INDEX.setdefault(loc, []).append(name)
```

---

## 5. Reliability & Error Handling

### REL-01: `SteeredCommittee.load_engine()` race condition [HIGH]

**File:** `committee.py:338-341`
**Effort:** 1 hour

**Description:** The `load_engine()` method checks `self._load_attempted` inside a lock but sets it to `True` before the engine is fully loaded. If loading fails, subsequent calls return `None` without retrying because `_load_attempted` is already `True`.

**Code (current):**
```python
# committee.py:340-343
async with self._load_lock:
    if self._engine is not None or self._load_attempted:
        return self._engine
    self._load_attempted = True
    try:
        ...
        self._engine = SteeringEngine()
```

**Suggested fix:** Only set `_load_attempted` on success, or add retry logic:
```python
async with self._load_lock:
    if self._engine is not None:
        return self._engine
    if self._load_attempted:
        return None  # Already failed once; or implement retry with backoff
    try:
        ...
        self._engine = SteeringEngine()
        self._engine.load_model()
        self._load_attempted = True  # Set AFTER success
    except Exception as e:
        logger.error(f"SteeredCommittee: failed to load engine: {e}")
        self._load_attempted = True  # Prevent infinite retry loops
        self._engine = None
```

---

### REL-02: LLM retry loop can block event loop 14+ minutes [HIGH]

**Files:** `llm.py:102-128`, `committee.py:273-300`
**Effort:** 3 hours

**Description:** Both `OllamaClient._generate_direct()` and `Committee._call_model()` retry up to 10 times with delays of 15s, 30s, 45s, ... up to 120s. While the `asyncio.sleep()` yields the event loop, the overall call can take over 14 minutes before giving up. Any agent waiting for this call is effectively stalled.

**Code (current):**
```python
# llm.py:102-128
max_retries = 10
base_delay = 15
for attempt in range(max_retries):
    try:
        ...
    except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
        delay = min(base_delay * (attempt + 1), 120)
        await asyncio.sleep(delay)
```

**Suggested fix:** Implement a circuit breaker pattern:
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def should_allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open: allow one attempt
```

Also reduce `max_retries` to 3-5 and cap total wait time.

---

### REL-03: Non-atomic state file writes [MEDIUM]

**File:** `main.py:637-643`
**Effort:** 30 minutes

**Description:** `_save_state()` writes directly to `latest_state.json`. If the process crashes mid-write, the file is corrupted and auto-resume fails.

**Code (current):**
```python
# main.py:641-643
latest_save = "saves/latest_state.json"
with open(latest_save, 'w') as f:
    json.dump(state, f, indent=2)
```

The same pattern exists in `telegram_broadcaster.py:293-295`:
```python
def save_state(offset: int):
    with open(STATE_FILE, "w") as f:
        json.dump({"offset": offset, ...}, f)
```

**Suggested fix:** Write to a temp file then atomic rename:
```python
import tempfile

tmp_fd, tmp_path = tempfile.mkstemp(dir="saves", suffix=".json")
try:
    with os.fdopen(tmp_fd, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, latest_save)  # Atomic on POSIX
except Exception:
    os.unlink(tmp_path)
    raise
```

---

### REL-04: Bare `except Exception: pass` silencing real errors [MEDIUM]

**Files:** `webui.py:59`, `llm.py:43`, `main.py:98`, and 7+ other locations
**Effort:** 1 hour

**Description:** Over 10 instances of `except Exception: pass` (or equivalent) swallow errors silently. This makes debugging extremely difficult when things go wrong.

**Examples:**
```python
# webui.py:57-60
try:
    await self.broadcast_tick()
except Exception:
    pass  # Silenced!

# llm.py:42-44
try:
    _llm_status_callback(agent, task, model)
except Exception:
    pass  # Silenced!
```

**Suggested fix:** At minimum, log at DEBUG level:
```python
except Exception:
    logger.debug("broadcast_tick failed", exc_info=True)
```

---

### REL-05: Qwen3 thinking block stripping can lose entire response [MEDIUM]

**File:** `steering/engine.py:215-226`
**Effort:** 30 minutes

**Description:** When Qwen3's `<think>` block is unclosed (truncated by `max_new_tokens`), the regex `<think>.*` with `re.DOTALL` removes everything from `<think>` to the end of the string, potentially discarding useful content that appeared before the thinking block.

**Code (current):**
```python
# steering/engine.py:215-226
text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
if '<think>' in text:
    logger.warning(f"Unclosed <think> block detected...")
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)

result = text.strip()
if not result:
    logger.warning(f"Empty result after stripping think blocks.")
```

**Suggested fix:** Preserve content before the unclosed `<think>` tag:
```python
if '<think>' in text:
    # Keep everything before the unclosed <think> block
    before_think = text[:text.index('<think>')].strip()
    if before_think:
        text = before_think
    else:
        logger.warning("No content before unclosed <think> block")
        text = ""
```

---

### REL-06: Telegram broadcaster uses blocking `requests.post()` [MEDIUM]

**File:** `telegram_broadcaster.py:80`
**Effort:** N/A (standalone script, not in async loop)

**Description:** `send_telegram()` uses the synchronous `requests` library. This is acceptable because the broadcaster runs as a separate process (not inside the async simulation loop), but it would block if integrated.

**Note:** This is informational. No fix needed unless the broadcaster is merged into the main simulation process.

---

### REL-07: Composite direction cache grows unbounded [LOW]

**File:** `steering/engine.py:38`
**Effort:** 30 minutes

**Description:** `_composite_cache` stores computed composite directions keyed by `"agent|role"`. With 25 agents × 5 pipeline roles = 125 entries, each containing large tensors, this can consume significant GPU memory over time.

**Suggested fix:** Add an LRU bound or use `functools.lru_cache`:
```python
from collections import OrderedDict

MAX_CACHE_SIZE = 150

def _get_composite_directions(self, agent_concepts, cache_key=""):
    if cache_key and cache_key in self._composite_cache:
        self._composite_cache.move_to_end(cache_key)  # LRU refresh
        return self._composite_cache[cache_key]
    ...
    if len(self._composite_cache) > MAX_CACHE_SIZE:
        self._composite_cache.popitem(last=False)  # Evict oldest
```

---

## 6. Security

### SEC-01: No authentication on WebUI REST API [CRITICAL]

**File:** `webui.py:82-93`
**Effort:** 1-3 hours

**Description:** All REST endpoints (`/api/agent/{name}`, `/api/pause`, `/api/resume`, etc.) and the WebSocket handler are completely unauthenticated. Anyone who can reach the server can pause/resume the simulation, read all agent data, and trigger state saves.

**Code (current):**
```python
# webui.py:82-93
def _register_routes(self):
    self.app.router.add_get("/", self.handle_index)
    self.app.router.add_get("/ws", self.ws_handler)
    self.app.router.add_get("/api/agent/{name}", self.handle_agent)
    ...
    self.app.router.add_post("/api/pause", self.handle_pause)
    self.app.router.add_post("/api/resume", self.handle_resume)
```

**Suggested fix (minimal):** Add a shared secret via environment variable:
```python
API_TOKEN = os.getenv("WEBUI_API_TOKEN", "")

@web.middleware
async def auth_middleware(request, handler):
    if API_TOKEN and request.path.startswith("/api"):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token != API_TOKEN:
            raise web.HTTPUnauthorized(text="Invalid API token")
    return await handler(request)
```

---

### SEC-02: WebUI binds to `0.0.0.0` (all interfaces) [HIGH]

**File:** `webui.py:46`
**Effort:** 15 minutes

**Description:** The web server binds to `0.0.0.0`, exposing it to all network interfaces. Combined with SEC-01 (no auth), anyone on the local network can control the simulation.

**Code (current):**
```python
# webui.py:46
self.site = web.TCPSite(self.runner, "0.0.0.0", port)
```

**Suggested fix:**
```python
host = os.getenv("WEBUI_HOST", "127.0.0.1")
self.site = web.TCPSite(self.runner, host, port)
```

---

## 7. Testing

### TEST-01: Zero test coverage for 5 core modules [HIGH]

**Files:** `llm.py`, `committee.py`, `webui.py`, `display.py`, `telegram_broadcaster.py`
**Effort:** 8-12 hours

**Description:** The `tests/` directory covers `memory.py`, `agent.py`, `environment.py`, and `conversation.py`, but has zero coverage for:

- **`llm.py`** — LLM client, retry logic, model routing
- **`committee.py`** — Expert pipeline orchestration, steered committee
- **`webui.py`** — REST API handlers, WebSocket lifecycle
- **`display.py`** — Terminal UI rendering
- **`telegram_broadcaster.py`** — Log parsing, Telegram integration

**Priority order for new tests:**
1. `llm.py` — Mock Ollama responses, test retry/backoff, test model routing
2. `committee.py` — Mock model calls, test pipeline sequencing, test expert prompt building
3. `webui.py` — Use `aiohttp.test_utils.AioHTTPTestCase`, test REST endpoints
4. `telegram_broadcaster.py` — Test regex parsing, digest formatting
5. `display.py` — Test data formatting methods (skip Rich rendering)

---

## 8. Code Quality

### CQ-01: Global singletons hinder testability [MEDIUM]

**Files:** `llm.py:215`, `committee.py:496`
**Effort:** 4 hours

**Description:** `llm_client` and `_committee` are module-level globals initialized on first use. This makes unit testing difficult because there's no clean way to inject mocks without monkey-patching.

**Code (current):**
```python
# llm.py:215
llm_client = OllamaClient()

# committee.py:496
_committee: Optional[Committee] = None
```

**Suggested fix:** Use dependency injection or a simple registry pattern:
```python
class LLMRegistry:
    _instance: Optional[OllamaClient] = None

    @classmethod
    def get(cls, use_gpu_queue=True) -> OllamaClient:
        if cls._instance is None:
            cls._instance = OllamaClient(use_gpu_queue)
        return cls._instance

    @classmethod
    def set(cls, client: OllamaClient):
        """For testing: inject a mock client."""
        cls._instance = client

    @classmethod
    def reset(cls):
        cls._instance = None
```

---

### CQ-02: Token budget unconditionally inflated for Qwen models [LOW]

**File:** `steering/engine.py:231-232`
**Effort:** 15 minutes

**Description:** Both `_generate_plain()` and `_generate_steered()` inflate `max_new_tokens` to 2048 when the model ID contains "Qwen", regardless of the actual task. Short tasks (importance scoring, yes/no decisions) waste compute generating unnecessary thinking tokens.

**Code (current):**
```python
# steering/engine.py:231-232
if "Qwen" in MODEL_ID and max_new_tokens < 1024:
    max_new_tokens = 2048
```

**Suggested fix:** Only inflate for tasks that genuinely need thinking room:
```python
# Only inflate for tasks that need thinking room (planning, reflection)
# Short tasks like importance scoring or yes/no should keep their budget
if "Qwen" in MODEL_ID and max_new_tokens >= 100 and max_new_tokens < 1024:
    max_new_tokens = 2048
```

---

### CQ-03: No log rotation [LOW]

**File:** `main.py:25-32`
**Effort:** 15 minutes

**Description:** `simulation.log` is opened in append mode with no rotation. Multi-day simulations can produce gigabytes of logs.

**Code (current):**
```python
# main.py:25-32
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
```

**Suggested fix:**
```python
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('simulation.log', maxBytes=50_000_000, backupCount=3),
        logging.StreamHandler(sys.stdout)
    ]
)
```

---

### CQ-04: Duplicate import of `IMPORTANCE_THRESHOLD` [LOW]

**File:** `reflection_engine.py:12`
**Effort:** 1 minute

**Description:** `IMPORTANCE_THRESHOLD` is imported twice on the same line.

**Code (current):**
```python
# reflection_engine.py:12
from config import IMPORTANCE_THRESHOLD, MAX_RECENT_MEMORIES, IMPORTANCE_THRESHOLD
```

**Suggested fix:**
```python
from config import IMPORTANCE_THRESHOLD, MAX_RECENT_MEMORIES
```

---

## 9. Quick Wins (Top 7)

These are the highest-impact changes achievable in under 1 hour each:

| Priority | ID | Fix | Impact | Effort |
|----------|----|-----|--------|--------|
| 1 | PERF-01 | Wrap `ollama_query()` in `run_in_executor()` | Unblocks entire event loop | 1h |
| 2 | DATA-03 | Add `PRAGMA journal_mode=WAL` | Eliminates write stalls | 30m |
| 3 | SEC-02 | Bind WebUI to `127.0.0.1` | Closes network exposure | 15m |
| 4 | DATA-06 | Add `timeout=30` to `sqlite3.connect()` | Prevents lock errors | 15m |
| 5 | CQ-03 | Add `RotatingFileHandler` | Prevents disk fill | 15m |
| 6 | PERF-04 | `asyncio.gather()` for planning + memo thoughts | 2x faster post-conversation | 30m |
| 7 | CQ-02 | Fix token budget inflation threshold | Saves compute on short tasks | 15m |

**Total quick win effort: ~2h 45m** for significant gains across performance, reliability, and security.

---

## 10. Architecture Notes

### Structural Strengths

- **Async-first design**: The entire simulation loop and agent cognition use `async/await`, enabling concurrent agent processing
- **Modular separation**: Clean boundaries between agent cognition, memory, environment, conversation, and display
- **FTS5 integration**: Full-text search for memory retrieval with triggers for automatic sync
- **Batch operations**: Memory access time updates and skill use count increments already use batch queries
- **Strategy pattern**: `ReflectionEngine`/`PlanningEngine`/`ConversationEngine` abstract away committee vs single-model modes
- **Rule-based importance scoring**: Avoids LLM calls for common observations via keyword matching

### Areas for Future Refactoring

- **Dependency injection**: Replace global singletons (`llm_client`, `_committee`) with injected dependencies for testability
- **Display/state decoupling**: `_update_display_data()` and `_update_stats()` in `main.py` mix data gathering with presentation; extract a `SimulationState` class
- **Circuit breaker pattern**: Replace retry loops with a proper circuit breaker to fail fast when Ollama is down
- **Connection management**: Centralize SQLite connection handling (WAL, timeout, pooling) in a shared database module
- **Incremental TF-IDF**: Both `MemoryStream` and `SkillBank` independently manage TF-IDF vectorizers; extract a shared `IncrementalVectorizer` class
