// Generative Agents Web Dashboard - Main Application
// Vanilla JavaScript (no frameworks)

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const state = {
    connected: false,
    paused: false,
    speed: 10,
    selectedAgent: null,
    agents: {},
    locations: {},
    conversations: [],
    events: [],
    stats: {},
    memoryPage: 0,
    memoryType: 'all',
    currentTime: null,
    tickCount: 0,
    day: 0
};

let ws = null;
let reconnectTimeout = null;

// ============================================================================
// WEBSOCKET CONNECTION
// ============================================================================

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log('Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        state.connected = true;
        updateConnectionStatus(true);
        clearTimeout(reconnectTimeout);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'tick') {
                handleTick(data);
            }
        } catch (err) {
            console.error('Error parsing WebSocket message:', err);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        state.connected = false;
        updateConnectionStatus(false);
        // Auto-reconnect after 3 seconds
        reconnectTimeout = setTimeout(connect, 3000);
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        state.connected = false;
        updateConnectionStatus(false);
    };
}

function updateConnectionStatus(connected) {
    const dotEl = document.getElementById('connection-status');
    const textEl = dotEl ? dotEl.parentElement.querySelector('.status-text') : null;
    if (dotEl) {
        dotEl.className = connected ? 'status-dot connected' : 'status-dot disconnected';
    }
    if (textEl) {
        textEl.textContent = connected ? 'Connected' : 'Disconnected';
    }
}

// ============================================================================
// TICK HANDLER
// ============================================================================

function handleTick(data) {
    // Update state
    state.currentTime = data.time;
    state.tickCount = data.tick_count;
    state.speed = data.speed;
    state.paused = !data.running;
    state.day = data.day;
    state.agents = data.agents || {};
    state.locations = data.locations || {};
    state.conversations = data.active_conversations || [];
    state.stats = data.stats || {};

    // Replace events with server's latest snapshot (server sends last 30)
    if (data.recent_events) {
        state.events = data.recent_events;
    }

    // Update header
    updateHeader(data);

    // Update events feed
    renderEventsFeed(state.events);

    // Update conversations panel
    renderConversations(state.conversations);

    // Update stats
    renderStats(state.stats, data.llm);

    // Update map (from map.js)
    if (window.updateMap) {
        window.updateMap(state.agents, state.locations);
    }

    // If agent selected, update their activity display
    if (state.selectedAgent && state.agents[state.selectedAgent]) {
        updateAgentActivity(state.selectedAgent, state.agents[state.selectedAgent]);
    }
}

function updateHeader(data) {
    // Update time display
    const timeEl = document.getElementById('sim-time');
    timeEl.textContent = formatTime(data.time);

    // Update speed display
    const speedEl = document.getElementById('speed-display');
    speedEl.textContent = `${data.speed}x`;

    // Update pause button
    const pauseBtn = document.getElementById('pause-btn');
    pauseBtn.textContent = data.running ? '⏸ Pause' : '▶ Resume';
}

// ============================================================================
// CONTROL HANDLERS
// ============================================================================

function togglePause() {
    if (!state.connected) return;

    const command = state.paused ? 'resume' : 'pause';
    ws.send(JSON.stringify({ type: command }));
}

function changeSpeed(delta) {
    if (!state.connected) return;

    const newSpeed = Math.max(1, Math.min(100, state.speed + delta));
    ws.send(JSON.stringify({ type: 'set_speed', speed: newSpeed }));
}

function saveState() {
    if (!state.connected) return;

    ws.send(JSON.stringify({ type: 'save_state' }));
    showToast('State saved!');
}

function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    // Fade in
    setTimeout(() => toast.classList.add('show'), 10);

    // Remove after 2 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

// ============================================================================
// AGENT DETAIL PANEL
// ============================================================================

async function selectAgent(agentName) {
    state.selectedAgent = agentName;
    state.memoryPage = 0;

    // Show loading state
    const placeholder = document.getElementById('agent-panel-placeholder');
    const detailPanel = document.getElementById('agent-detail');
    placeholder.style.display = 'none';
    detailPanel.style.display = 'block';

    // Highlight on map
    if (window.highlightAgent) {
        window.highlightAgent(agentName);
    }

    try {
        // Fetch full agent details
        const response = await fetch(`/api/agent/${encodeURIComponent(agentName)}`);
        if (!response.ok) throw new Error('Failed to fetch agent details');

        const data = await response.json();
        populateAgentDetail(data);

        // Load initial memories
        await loadMemories(agentName, state.memoryType, 0);
    } catch (err) {
        console.error('Error loading agent details:', err);
        showToast('Error loading agent details');
    }
}

function clearAgentSelection() {
    state.selectedAgent = null;
    state.memoryPage = 0;

    const placeholder = document.getElementById('agent-panel-placeholder');
    const detailPanel = document.getElementById('agent-detail');
    placeholder.style.display = 'flex';
    detailPanel.style.display = 'none';

    // Clear map highlight
    if (window.clearHighlight) {
        window.clearHighlight();
    }
}

function populateAgentDetail(data) {
    const persona = data.persona || {};

    // Name and meta
    document.getElementById('agent-name').textContent = data.name;
    document.getElementById('agent-meta').textContent =
        `${persona.age || '?'} years old, ${persona.occupation || 'Unknown'}`;

    // Personality
    document.getElementById('agent-personality').textContent = persona.personality || '';

    // Current activity
    updateAgentActivity(data.name, state.agents[data.name] || {
        activity: data.current_activity,
        location: data.current_location,
        sub_area: data.current_sub_area
    });

    // Daily schedule (from plan items or persona routine)
    const scheduleEl = document.getElementById('agent-schedule');
    if (data.daily_plan && data.daily_plan.length > 0) {
        scheduleEl.innerHTML = data.daily_plan
            .map(item => {
                const time = new Date(item.start_time);
                const hh = String(time.getHours()).padStart(2, '0');
                const mm = String(time.getMinutes()).padStart(2, '0');
                const done = item.completed ? ' ✓' : '';
                return `<li><span class="schedule-time">${hh}:${mm}</span> ${escapeHtml(item.description)}${done}</li>`;
            })
            .join('');
    } else if (persona.daily_routine && persona.daily_routine.length > 0) {
        scheduleEl.innerHTML = persona.daily_routine
            .map(item => `<li>${escapeHtml(item)}</li>`)
            .join('');
    } else {
        scheduleEl.innerHTML = '<li>No schedule defined</li>';
    }

    // Relationships
    const relsEl = document.getElementById('agent-relationships');
    if (persona.relationships && Object.keys(persona.relationships).length > 0) {
        relsEl.innerHTML = Object.entries(persona.relationships)
            .map(([name, desc]) => `<li><strong>${escapeHtml(name)}</strong>: ${escapeHtml(desc)}</li>`)
            .join('');
    } else {
        relsEl.innerHTML = '<li>No relationships defined</li>';
    }
}

function updateAgentActivity(agentName, agentData) {
    if (state.selectedAgent !== agentName) return;

    const activityEl = document.getElementById('agent-activity');
    if (agentData.activity && agentData.location) {
        const subArea = agentData.sub_area ? ` (${agentData.sub_area})` : '';
        const llmStatus = agentData.llm_active ? ' 🧠' : '';
        activityEl.textContent = `${agentData.activity} at ${agentData.location}${subArea}${llmStatus}`;
    } else {
        activityEl.textContent = 'Activity unknown';
    }
}

// ============================================================================
// MEMORY BROWSER
// ============================================================================

async function loadMemories(agentName, type = 'all', offset = 0) {
    try {
        const params = new URLSearchParams({
            type: type,
            limit: 20,
            offset: offset
        });

        const response = await fetch(`/api/agent/${encodeURIComponent(agentName)}/memories?${params}`);
        if (!response.ok) throw new Error('Failed to fetch memories');

        const result = await response.json();
        const memories = result.memories || [];

        const container = document.getElementById('agent-memories');

        // If first page, clear container
        if (offset === 0) {
            container.innerHTML = '';
        }

        // Render memory items
        if (memories.length === 0 && offset === 0) {
            container.innerHTML = '<div class="memory-item">No memories found</div>';
        } else {
            memories.forEach(memory => {
                container.appendChild(createMemoryElement(memory));
            });
        }

        // Update load more button
        const loadMoreBtn = document.getElementById('load-more-memories');
        if (memories.length >= 20) {
            loadMoreBtn.style.display = 'block';
            state.memoryPage = offset + 20;
        } else {
            loadMoreBtn.style.display = 'none';
        }
    } catch (err) {
        console.error('Error loading memories:', err);
        showToast('Error loading memories');
    }
}

function createMemoryElement(memory) {
    const div = document.createElement('div');
    div.className = 'memory-item';

    const typeColors = {
        observation: '#3498db',
        reflection: '#9b59b6',
        plan: '#27ae60'
    };

    const memType = memory.memory_type || 'observation';
    const badgeColor = typeColors[memType] || '#95a5a6';

    div.innerHTML = `
        <div class="memory-header">
            <span class="memory-type" style="background-color: ${badgeColor}">${memType}</span>
            <span class="memory-importance">★ ${memory.importance_score || 0}</span>
        </div>
        <div class="memory-description">${escapeHtml(memory.description)}</div>
        <div class="memory-meta">
            ${formatTime(memory.creation_timestamp)} • ${escapeHtml(memory.location || 'Unknown')}
        </div>
    `;

    return div;
}

function setMemoryFilter(type) {
    state.memoryType = type;
    state.memoryPage = 0;

    // Update active tab
    document.querySelectorAll('.filter-tab').forEach(tab => {
        if (tab.dataset.filter === type) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    // Reload memories
    if (state.selectedAgent) {
        loadMemories(state.selectedAgent, type, 0);
    }
}

function loadMoreMemories() {
    if (state.selectedAgent) {
        loadMemories(state.selectedAgent, state.memoryType, state.memoryPage);
    }
}

// ============================================================================
// UI RENDERING
// ============================================================================

function renderEventsFeed(events) {
    const container = document.getElementById('events-feed');

    // Keep only last 30 events for display
    const displayEvents = events.slice(-30);

    // Check if we should auto-scroll (user is at bottom)
    const shouldAutoScroll = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;

    container.innerHTML = displayEvents
        .map(event => `<div class="event-item">${escapeHtml(event)}</div>`)
        .join('');

    // Auto-scroll to bottom if user was already at bottom
    if (shouldAutoScroll) {
        container.scrollTop = container.scrollHeight;
    }
}

function renderConversations(activeConvos) {
    const activeContainer = document.getElementById('active-conversations');
    if (!activeContainer) return;

    if (activeConvos.length === 0) {
        activeContainer.innerHTML = '<div class="empty-state">No active conversations</div>';
        return;
    }

    activeContainer.innerHTML = activeConvos
        .map(convo => `
            <div class="conversation-item" data-agent1="${escapeHtml(convo.agent1)}" data-agent2="${escapeHtml(convo.agent2)}">
                <div class="conversation-participants">
                    <strong>${escapeHtml(convo.agent1)}</strong> & <strong>${escapeHtml(convo.agent2)}</strong>
                </div>
                <div class="conversation-meta">
                    ${escapeHtml(convo.location)} • ${convo.turns} turns
                </div>
            </div>
        `)
        .join('');

    // Add click handlers to show conversation transcripts
    activeContainer.querySelectorAll('.conversation-item').forEach(item => {
        item.addEventListener('click', () => {
            showConversationTranscript(item.dataset.agent1, item.dataset.agent2);
        });
    });
}

async function showConversationTranscript(agent1, agent2) {
    try {
        const response = await fetch(`/api/conversations`);
        if (!response.ok) throw new Error('Failed to fetch conversations');

        const result = await response.json();
        const allConvos = result.conversations || [];
        const convo = allConvos.find(c =>
            (c.agent1 === agent1 && c.agent2 === agent2) ||
            (c.agent1 === agent2 && c.agent2 === agent1)
        );

        if (!convo || !convo.turns || convo.turns.length === 0) {
            showToast('No transcript available');
            return;
        }

        // Create modal overlay
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Conversation: ${escapeHtml(agent1)} & ${escapeHtml(agent2)}</h3>
                    <button class="close-btn">&times;</button>
                </div>
                <div class="modal-body">
                    ${convo.turns.map(turn => `
                        <div class="transcript-turn">
                            <strong>${escapeHtml(turn.speaker)}:</strong> ${escapeHtml(turn.message)}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close handlers
        modal.querySelector('.close-btn').addEventListener('click', () => modal.remove());
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
    } catch (err) {
        console.error('Error loading conversation:', err);
        showToast('Error loading conversation');
    }
}

function renderStats(stats, llm) {
    // Update stat counters
    const statElements = {
        'stat-memories': stats.total_memories || 0,
        'stat-conversations': stats.total_conversations || 0,
        'stat-reflections': stats.total_reflections || 0,
        'stat-observations': stats.total_observations || 0,
        'stat-tick': state.tickCount,
        'stat-llm-calls': llm ? llm.call_count || 0 : 0
    };

    Object.entries(statElements).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value.toLocaleString();
        }
    });

    // Update LLM status if present
    if (llm && llm.agent) {
        const llmStatus = document.getElementById('llm-status');
        if (llmStatus) {
            llmStatus.textContent = `${llm.agent} (${llm.task}) - ${llm.model}`;
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function formatTime(isoString) {
    if (!isoString) return 'Unknown';

    const date = new Date(isoString);
    const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    const dayName = days[date.getDay()];
    const monthName = months[date.getMonth()];
    const day = date.getDate();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');

    return `${dayName}, ${monthName} ${day} ${hours}:${minutes}`;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// MAP INTEGRATION CALLBACKS
// ============================================================================

// Set up callbacks for map.js to call
window.onAgentClick = (agentName) => {
    selectAgent(agentName);
};

window.onBuildingHover = (buildingName, agents) => {
    // Optional: show tooltip with agents in building
    console.log(`Hovering over ${buildingName}:`, agents);
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Generative Agents Dashboard');

    // Initialize map
    if (window.initMap) {
        window.initMap('map-canvas');
    }

    // Connect to WebSocket
    connect();

    // Bind control buttons
    const pauseBtn = document.getElementById('pause-btn');
    pauseBtn.addEventListener('click', togglePause);

    const saveBtn = document.getElementById('save-btn');
    saveBtn.addEventListener('click', saveState);

    const speedDown = document.getElementById('speed-decrease');
    if (speedDown) speedDown.addEventListener('click', () => changeSpeed(-1));

    const speedUp = document.getElementById('speed-increase');
    if (speedUp) speedUp.addEventListener('click', () => changeSpeed(1));

    // Bind memory filter tabs
    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            setMemoryFilter(tab.dataset.filter);
        });
    });

    // Bind load more memories button
    const loadMoreBtn = document.getElementById('load-more-memories');
    loadMoreBtn.addEventListener('click', loadMoreMemories);

    // Close agent detail panel when clicking the X
    const closeBtn = document.querySelector('.close-agent-panel');
    if (closeBtn) {
        closeBtn.addEventListener('click', clearAgentSelection);
    }

    console.log('Dashboard initialized');
});
