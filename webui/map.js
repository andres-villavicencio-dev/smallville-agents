// map.js - HTML5 Canvas pixel-art map renderer for Smallville simulation

let canvas, ctx;
let canvasWidth, canvasHeight;
let scale = 1;

// Building definitions
let buildings = [];

// Agent rendering
const agentColors = {};
let agentPositions = {};
let highlightedAgent = null;

// Animation
let animationFrame = null;

// Tooltip element
let tooltip = null;

// Agent names in fixed order for color assignment
const AGENT_NAMES = [
    "John Lin", "Mei Lin", "Eddy Lin", "Isabella Rodriguez", "Tom Moreno",
    "Sam Moore", "Carmen Moreno", "Carlos Gomez", "Maria Santos", "Sarah Chen",
    "Mike Johnson", "Jennifer Moore", "Emily Moore", "Diego Moreno", "Ana Santos",
    "Dr. Williams", "Professor Anderson", "Professor Davis", "Lisa Park",
    "Mayor Johnson", "Miguel Rodriguez", "Mrs. Peterson", "Officer Thompson",
    "Rachel Kim", "Frank Wilson"
];

// Location name mapping (simulation uses these exact strings)
const LOCATION_MAPPING = {
    "Lin Family Home": "Lin Family Home",
    "Moreno Family Home": "Moreno Family Home",
    "Moore Family Home": "Moore Family Home",
    "The Willows": "The Willows",
    "Oak Hill College": "Oak Hill College",
    "Harvey Oak Supply Store": "Harvey Oak Supply Store",
    "The Rose and Crown Pub": "The Rose and Crown Pub",
    "Hobbs Cafe": "Hobbs Cafe",
    "Johnson Park": "Johnson Park",
    "Town Hall": "Town Hall",
    "Library": "Library",
    "Pharmacy": "Pharmacy"
};

function initMap(canvasId) {
    canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error('Canvas element not found:', canvasId);
        return;
    }

    ctx = canvas.getContext('2d');

    // Create tooltip element
    tooltip = document.createElement('div');
    tooltip.id = 'map-tooltip';
    tooltip.style.position = 'absolute';
    tooltip.style.background = '#1a1a2e';
    tooltip.style.color = 'white';
    tooltip.style.padding = '8px 12px';
    tooltip.style.borderRadius = '6px';
    tooltip.style.fontSize = '12px';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.display = 'none';
    tooltip.style.zIndex = '1000';
    tooltip.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
    document.body.appendChild(tooltip);

    // Assign agent colors
    assignAgentColors();

    // Setup responsive canvas
    handleResize();

    // Event listeners
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('click', handleClick);
    canvas.addEventListener('mouseleave', hideTooltip);
    window.addEventListener('resize', handleResize);

    // Start animation loop
    animate();
}

function assignAgentColors() {
    AGENT_NAMES.forEach((name, i) => {
        agentColors[name] = `hsl(${i * (360/25)}, 70%, 55%)`;
        agentPositions[name] = { x: 0, y: 0, targetX: 0, targetY: 0 };
    });
}

function handleResize() {
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;

    // Get container size
    const rect = container.getBoundingClientRect();
    canvasWidth = rect.width;
    canvasHeight = rect.height;

    // Set display size
    canvas.style.width = canvasWidth + 'px';
    canvas.style.height = canvasHeight + 'px';

    // Set actual size in memory (scaled for retina)
    scale = dpr;
    canvas.width = canvasWidth * dpr;
    canvas.height = canvasHeight * dpr;

    // Scale context
    ctx.scale(dpr, dpr);

    // Recalculate layout
    calculateLayout();
}

function calculateLayout() {
    buildings = [];

    // Grid layout: 3 rows, 4 columns
    const margin = 40;
    const roadWidth = 20;
    const cols = 4;
    const rows = 3;

    const availableWidth = canvasWidth - (2 * margin) - (roadWidth * (cols - 1));
    const availableHeight = canvasHeight - (2 * margin) - (roadWidth * (rows - 1));

    const buildingWidth = availableWidth / cols;
    const buildingHeight = availableHeight / rows;
    const roofHeight = buildingHeight * 0.25;

    // Row 1: Residential
    buildings.push({
        name: "Lin Family Home",
        x: margin,
        y: margin,
        w: buildingWidth * 0.8,
        h: buildingHeight * 0.7,
        color: "#8B6914",
        roofColor: "#654321",
        label: "Lin Home"
    });

    buildings.push({
        name: "Moreno Family Home",
        x: margin + (buildingWidth + roadWidth),
        y: margin,
        w: buildingWidth * 0.8,
        h: buildingHeight * 0.7,
        color: "#A0522D",
        roofColor: "#6B3410",
        label: "Moreno Home"
    });

    buildings.push({
        name: "Moore Family Home",
        x: margin + 2 * (buildingWidth + roadWidth),
        y: margin,
        w: buildingWidth * 0.8,
        h: buildingHeight * 0.7,
        color: "#CD853F",
        roofColor: "#8B5A2B",
        label: "Moore Home"
    });

    buildings.push({
        name: "The Willows",
        x: margin + 3 * (buildingWidth + roadWidth),
        y: margin,
        w: buildingWidth * 0.85,
        h: buildingHeight * 0.75,
        color: "#DEB887",
        roofColor: "#BC9162",
        label: "The Willows"
    });

    // Row 2: Commercial/Public
    const row2Y = margin + buildingHeight + roadWidth;

    buildings.push({
        name: "Library",
        x: margin,
        y: row2Y,
        w: buildingWidth * 0.85,
        h: buildingHeight * 0.75,
        color: "#708090",
        roofColor: "#556677",
        label: "Library"
    });

    buildings.push({
        name: "Town Hall",
        x: margin + (buildingWidth + roadWidth),
        y: row2Y,
        w: buildingWidth * 0.9,
        h: buildingHeight * 0.8,
        color: "#778899",
        roofColor: "#5F6B7A",
        label: "Town Hall"
    });

    buildings.push({
        name: "Pharmacy",
        x: margin + 2 * (buildingWidth + roadWidth),
        y: row2Y,
        w: buildingWidth * 0.75,
        h: buildingHeight * 0.7,
        color: "#5F9EA0",
        roofColor: "#4A7A7C",
        label: "Pharmacy"
    });

    buildings.push({
        name: "Harvey Oak Supply Store",
        x: margin + 3 * (buildingWidth + roadWidth),
        y: row2Y,
        w: buildingWidth * 0.85,
        h: buildingHeight * 0.75,
        color: "#4682B4",
        roofColor: "#36648B",
        label: "Supply Store"
    });

    // Row 3: Social
    const row3Y = margin + 2 * (buildingHeight + roadWidth);

    buildings.push({
        name: "The Rose and Crown Pub",
        x: margin,
        y: row3Y,
        w: buildingWidth * 0.85,
        h: buildingHeight * 0.75,
        color: "#B22222",
        roofColor: "#8B1A1A",
        label: "Pub"
    });

    buildings.push({
        name: "Hobbs Cafe",
        x: margin + (buildingWidth + roadWidth),
        y: row3Y,
        w: buildingWidth * 0.8,
        h: buildingHeight * 0.7,
        color: "#D2691E",
        roofColor: "#A0521E",
        label: "Cafe"
    });

    buildings.push({
        name: "Johnson Park",
        x: margin + 2 * (buildingWidth + roadWidth),
        y: row3Y,
        w: buildingWidth * 0.9,
        h: buildingHeight * 0.75,
        color: "#228B22",
        roofColor: "#1B6B1B",
        label: "Park"
    });

    buildings.push({
        name: "Oak Hill College",
        x: margin + 3 * (buildingWidth + roadWidth),
        y: row3Y,
        w: buildingWidth * 0.9,
        h: buildingHeight * 0.8,
        color: "#4169E1",
        roofColor: "#2F4F9F",
        label: "College"
    });
}

function drawMap() {
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    drawGround();
    drawRoads();

    buildings.forEach(building => {
        drawBuilding(building);
    });

    drawDecorations();
}

function drawGround() {
    // Base green
    ctx.fillStyle = '#2d5a27';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Add grass texture
    ctx.fillStyle = '#264521';
    for (let i = 0; i < 1000; i++) {
        const x = Math.random() * canvasWidth;
        const y = Math.random() * canvasHeight;
        const size = Math.random() * 2;
        ctx.fillRect(x, y, size, size);
    }

    ctx.fillStyle = '#336633';
    for (let i = 0; i < 800; i++) {
        const x = Math.random() * canvasWidth;
        const y = Math.random() * canvasHeight;
        const size = Math.random() * 2;
        ctx.fillRect(x, y, size, size);
    }
}

function drawRoads() {
    ctx.fillStyle = '#666666';

    const margin = 40;
    const roadWidth = 20;

    // Horizontal roads
    const buildingHeight = (canvasHeight - 2 * margin - 2 * roadWidth) / 3;

    // Road after row 1
    ctx.fillRect(0, margin + buildingHeight, canvasWidth, roadWidth);

    // Road after row 2
    ctx.fillRect(0, margin + 2 * buildingHeight + roadWidth, canvasWidth, roadWidth);

    // Center lines
    ctx.strokeStyle = '#888888';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    ctx.beginPath();
    ctx.moveTo(0, margin + buildingHeight + roadWidth / 2);
    ctx.lineTo(canvasWidth, margin + buildingHeight + roadWidth / 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, margin + 2 * buildingHeight + roadWidth + roadWidth / 2);
    ctx.lineTo(canvasWidth, margin + 2 * buildingHeight + roadWidth + roadWidth / 2);
    ctx.stroke();

    ctx.setLineDash([]);

    // Vertical roads between columns
    const buildingWidth = (canvasWidth - 2 * margin - 3 * roadWidth) / 4;

    for (let i = 0; i < 3; i++) {
        const x = margin + (i + 1) * buildingWidth + i * roadWidth;
        ctx.fillRect(x, 0, roadWidth, canvasHeight);
    }
}

function drawBuilding(building) {
    const { x, y, w, h, color, roofColor, label } = building;

    // Building body
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);

    // Roof (triangle)
    ctx.fillStyle = roofColor;
    ctx.beginPath();
    ctx.moveTo(x - 5, y);
    ctx.lineTo(x + w / 2, y - h * 0.25);
    ctx.lineTo(x + w + 5, y);
    ctx.closePath();
    ctx.fill();

    // Windows (2x2 grid)
    ctx.fillStyle = '#FFE4B5';
    const windowSize = Math.min(w * 0.15, 12);
    const windowSpacingX = w / 3;
    const windowSpacingY = h / 3;

    for (let row = 0; row < 2; row++) {
        for (let col = 0; col < 2; col++) {
            const wx = x + windowSpacingX * (col + 0.7);
            const wy = y + windowSpacingY * (row + 0.5);
            ctx.fillRect(wx, wy, windowSize, windowSize);

            // Window pane divider
            ctx.strokeStyle = '#D4A574';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(wx + windowSize / 2, wy);
            ctx.lineTo(wx + windowSize / 2, wy + windowSize);
            ctx.moveTo(wx, wy + windowSize / 2);
            ctx.lineTo(wx + windowSize, wy + windowSize / 2);
            ctx.stroke();
        }
    }

    // Door
    const doorWidth = w * 0.2;
    const doorHeight = h * 0.3;
    const doorX = x + w / 2 - doorWidth / 2;
    const doorY = y + h - doorHeight;

    ctx.fillStyle = '#3E2723';
    ctx.fillRect(doorX, doorY, doorWidth, doorHeight);

    // Door knob
    ctx.fillStyle = '#FFD700';
    ctx.beginPath();
    ctx.arc(doorX + doorWidth * 0.75, doorY + doorHeight / 2, 2, 0, Math.PI * 2);
    ctx.fill();

    // Label
    ctx.fillStyle = 'white';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.strokeText(label, x + w / 2, y + h + 18);
    ctx.fillText(label, x + w / 2, y + h + 18);
}

function drawDecorations() {
    // Trees scattered around
    const trees = [
        { x: 30, y: 150, size: 15 },
        { x: canvasWidth - 40, y: 180, size: 18 },
        { x: 50, y: canvasHeight - 80, size: 12 },
        { x: canvasWidth - 60, y: canvasHeight - 100, size: 14 },
        { x: canvasWidth / 2 - 100, y: 50, size: 16 },
        { x: canvasWidth / 2 + 120, y: canvasHeight - 60, size: 13 }
    ];

    trees.forEach(tree => {
        // Trunk
        ctx.fillStyle = '#654321';
        ctx.fillRect(tree.x - 3, tree.y, 6, tree.size);

        // Canopy
        ctx.fillStyle = '#228B22';
        ctx.beginPath();
        ctx.arc(tree.x, tree.y - tree.size / 2, tree.size, 0, Math.PI * 2);
        ctx.fill();

        // Highlight
        ctx.fillStyle = '#32CD32';
        ctx.beginPath();
        ctx.arc(tree.x - tree.size / 3, tree.y - tree.size / 2 - tree.size / 4, tree.size / 3, 0, Math.PI * 2);
        ctx.fill();
    });

    // Pond near Johnson Park
    const parkBuilding = buildings.find(b => b.name === "Johnson Park");
    if (parkBuilding) {
        ctx.fillStyle = '#4682B4';
        ctx.beginPath();
        ctx.ellipse(
            parkBuilding.x + parkBuilding.w / 2,
            parkBuilding.y + parkBuilding.h / 2,
            parkBuilding.w * 0.3,
            parkBuilding.h * 0.25,
            0, 0, Math.PI * 2
        );
        ctx.fill();

        // Water highlight
        ctx.fillStyle = '#87CEEB';
        ctx.beginPath();
        ctx.ellipse(
            parkBuilding.x + parkBuilding.w / 2 - 10,
            parkBuilding.y + parkBuilding.h / 2 - 8,
            parkBuilding.w * 0.15,
            parkBuilding.h * 0.12,
            0, 0, Math.PI * 2
        );
        ctx.fill();
    }

    // Bushes near some buildings
    buildings.slice(0, 4).forEach(building => {
        ctx.fillStyle = '#2F4F2F';
        ctx.beginPath();
        ctx.arc(building.x - 8, building.y + building.h / 2, 6, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(building.x + building.w + 8, building.y + building.h / 2, 7, 0, Math.PI * 2);
        ctx.fill();
    });
}

function drawAgents(agents, locations) {
    if (!agents || !locations) return;

    // Group agents by location
    const agentsByLocation = {};

    agents.forEach(agent => {
        const location = locations[agent.name];
        if (!location) return;

        if (!agentsByLocation[location]) {
            agentsByLocation[location] = [];
        }
        agentsByLocation[location].push(agent.name);
    });

    // Draw agents at their buildings
    Object.entries(agentsByLocation).forEach(([location, agentNames]) => {
        const building = buildings.find(b => b.name === location);
        if (!building) return;

        // Spread agents in a grid within building bounds
        const numAgents = agentNames.length;
        const gridSize = Math.ceil(Math.sqrt(numAgents));
        const cellWidth = building.w / (gridSize + 1);
        const cellHeight = building.h / (gridSize + 1);

        agentNames.forEach((agentName, index) => {
            const row = Math.floor(index / gridSize);
            const col = index % gridSize;

            const targetX = building.x + cellWidth * (col + 1);
            const targetY = building.y + cellHeight * (row + 1);

            // Update target position
            if (agentPositions[agentName]) {
                agentPositions[agentName].targetX = targetX;
                agentPositions[agentName].targetY = targetY;
            }
        });
    });

    // Draw all agent dots
    Object.entries(agentPositions).forEach(([agentName, pos]) => {
        const color = agentColors[agentName];
        if (!color) return;

        // Draw highlighted ring
        if (highlightedAgent === agentName) {
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 3;
            ctx.shadowColor = 'white';
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 9, 0, Math.PI * 2);
            ctx.stroke();
            ctx.shadowBlur = 0;
        }

        // Draw agent dot
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
        ctx.fill();

        // Outline
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
        ctx.stroke();
    });
}

function animateAgents() {
    const speed = 0.1;

    Object.entries(agentPositions).forEach(([agentName, pos]) => {
        pos.x = lerp(pos.x, pos.targetX, speed);
        pos.y = lerp(pos.y, pos.targetY, speed);
    });
}

function lerp(current, target, speed) {
    return current + (target - current) * speed;
}

function updateMap(agents, locations) {
    drawAgents(agents, locations);
}

function animate() {
    animateAgents();
    drawMap();

    // Redraw agents with current positions
    Object.entries(agentPositions).forEach(([agentName, pos]) => {
        const color = agentColors[agentName];
        if (!color) return;

        // Draw highlighted ring
        if (highlightedAgent === agentName) {
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 3;
            ctx.shadowColor = 'white';
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 9, 0, Math.PI * 2);
            ctx.stroke();
            ctx.shadowBlur = 0;
        }

        // Draw agent dot
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
        ctx.fill();

        // Outline
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
        ctx.stroke();
    });

    animationFrame = requestAnimationFrame(animate);
}

function getHoveredBuilding(mouseX, mouseY) {
    for (let i = buildings.length - 1; i >= 0; i--) {
        const b = buildings[i];
        if (mouseX >= b.x && mouseX <= b.x + b.w &&
            mouseY >= b.y && mouseY <= b.y + b.h) {
            return b;
        }
    }
    return null;
}

function getClickedAgent(mouseX, mouseY) {
    const clickRadius = 10;

    for (const [agentName, pos] of Object.entries(agentPositions)) {
        const dx = mouseX - pos.x;
        const dy = mouseY - pos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance <= clickRadius) {
            return agentName;
        }
    }

    return null;
}

function showTooltip(building, mouseX, mouseY) {
    // Count agents in this building
    const agentsHere = [];

    Object.entries(agentPositions).forEach(([agentName, pos]) => {
        if (pos.targetX >= building.x && pos.targetX <= building.x + building.w &&
            pos.targetY >= building.y && pos.targetY <= building.y + building.h) {
            agentsHere.push(agentName);
        }
    });

    let content = `<strong>${building.name}</strong>`;
    if (agentsHere.length > 0) {
        content += `<br><small>${agentsHere.length} agent${agentsHere.length > 1 ? 's' : ''} here</small>`;
    }

    tooltip.innerHTML = content;
    tooltip.style.display = 'block';
    tooltip.style.left = (mouseX + 15) + 'px';
    tooltip.style.top = (mouseY + 15) + 'px';
}

function hideTooltip() {
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}

function highlightAgent(agentName) {
    highlightedAgent = agentName;
}

function clearHighlight() {
    highlightedAgent = null;
}

function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const building = getHoveredBuilding(mouseX, mouseY);
    const agent = getClickedAgent(mouseX, mouseY);

    if (building) {
        showTooltip(building, e.clientX, e.clientY);
        canvas.style.cursor = 'pointer';
    } else if (agent) {
        canvas.style.cursor = 'pointer';
        hideTooltip();
    } else {
        hideTooltip();
        canvas.style.cursor = 'default';
    }
}

function handleClick(e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Check for agent click first (more specific)
    const agent = getClickedAgent(mouseX, mouseY);
    if (agent && window.onAgentClick) {
        window.onAgentClick(agent);
        return;
    }

    // Then check building click
    const building = getHoveredBuilding(mouseX, mouseY);
    if (building) {
        // Could add building click handler here if needed
        console.log('Clicked building:', building.name);
    }
}

// Export functions for app.js
window.initMap = initMap;
window.updateMap = updateMap;
window.highlightAgent = highlightAgent;
window.clearHighlight = clearHighlight;
