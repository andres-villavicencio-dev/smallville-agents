// map.js - Phaser 3 pixel-art map renderer for Smallville simulation
// Replaces the raw Canvas implementation with proper RPG-style visualization

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================

const TILE_SIZE = 16;
const MAP_WIDTH = 60;  // tiles
const MAP_HEIGHT = 40; // tiles

// Agent names for color assignment
const AGENT_NAMES = [
    "John Lin", "Mei Lin", "Eddy Lin", "Isabella Rodriguez", "Tom Moreno",
    "Sam Moore", "Carmen Moreno", "Carlos Gomez", "Maria Santos", "Sarah Chen",
    "Mike Johnson", "Jennifer Moore", "Emily Moore", "Diego Moreno", "Ana Santos",
    "Dr. Williams", "Professor Anderson", "Professor Davis", "Lisa Park",
    "Mayor Johnson", "Miguel Rodriguez", "Mrs. Peterson", "Officer Thompson",
    "Rachel Kim", "Frank Wilson"
];

// Tile indices (must match generate_assets.html)
const TILES = {
    GRASS: 0, GRASS_DARK: 1, GRASS_FLOWERS: 2, DIRT: 3, DIRT_PATH: 4,
    ROAD_H: 5, ROAD_V: 6, ROAD_CROSS: 7,
    ROAD_CORNER_TL: 8, ROAD_CORNER_TR: 9, ROAD_CORNER_BL: 10, ROAD_CORNER_BR: 11,
    ROAD_T_UP: 12, ROAD_T_DOWN: 13, ROAD_T_LEFT: 14, ROAD_T_RIGHT: 15,
    WATER: 16, WATER_EDGE_T: 17, WATER_EDGE_B: 18, WATER_EDGE_L: 19, WATER_EDGE_R: 20,
    TREE_TOP: 21, TREE_BOTTOM: 22, BUSH: 23,
    FLOWERS_RED: 24, FLOWERS_YELLOW: 25, FLOWERS_BLUE: 26, ROCK: 27,
    FENCE_H: 28, FENCE_V: 29, FENCE_CORNER: 30, FENCE_POST: 31,
    WALL_BROWN_TL: 32, WALL_BROWN_T: 33, WALL_BROWN_TR: 34,
    WALL_BROWN_L: 35, WALL_BROWN_C: 36, WALL_BROWN_R: 37,
    WALL_BROWN_BL: 38, WALL_BROWN_B: 39, WALL_BROWN_BR: 40,
    DOOR_BROWN: 41, WINDOW_BROWN: 42,
    ROOF_BROWN_L: 43, ROOF_BROWN_C: 44, ROOF_BROWN_R: 45, ROOF_BROWN_PEAK: 46, CHIMNEY: 47,
    WALL_GRAY_TL: 48, WALL_GRAY_T: 49, WALL_GRAY_TR: 50,
    WALL_GRAY_L: 51, WALL_GRAY_C: 52, WALL_GRAY_R: 53,
    WALL_GRAY_BL: 54, WALL_GRAY_B: 55, WALL_GRAY_BR: 56,
    DOOR_GRAY: 57, WINDOW_GRAY: 58,
    ROOF_GRAY_L: 59, ROOF_GRAY_C: 60, ROOF_GRAY_R: 61, PILLAR: 62, ARCH: 63,
    WALL_RED_TL: 64, WALL_RED_T: 65, WALL_RED_TR: 66,
    WALL_RED_L: 67, WALL_RED_C: 68, WALL_RED_R: 69,
    WALL_RED_BL: 70, WALL_RED_B: 71, WALL_RED_BR: 72,
    DOOR_RED: 73, WINDOW_RED: 74,
    ROOF_RED_L: 75, ROOF_RED_C: 76, ROOF_RED_R: 77, SIGN_PUB: 78, SIGN_CAFE: 79,
    WALL_BLUE_TL: 80, WALL_BLUE_T: 81, WALL_BLUE_TR: 82,
    WALL_BLUE_L: 83, WALL_BLUE_C: 84, WALL_BLUE_R: 85,
    WALL_BLUE_BL: 86, WALL_BLUE_B: 87, WALL_BLUE_BR: 88,
    DOOR_BLUE: 89, WINDOW_BLUE: 90,
    ROOF_BLUE_L: 91, ROOF_BLUE_C: 92, ROOF_BLUE_R: 93, SIGN_COLLEGE: 94, SIGN_STORE: 95,
    PARK_GRASS: 96, PARK_BENCH: 97, PARK_LAMP: 98,
    FOUNTAIN_TL: 99, FOUNTAIN_TR: 100, FOUNTAIN_BL: 101, FOUNTAIN_BR: 102,
    PLAYGROUND: 103,
    POND_TL: 104, POND_TR: 105, POND_BL: 106, POND_BR: 107,
    BRIDGE_H: 108, BRIDGE_V: 109, GAZEBO_TOP: 110, GAZEBO_BOTTOM: 111,
    FLOOR_WOOD: 112, FLOOR_TILE: 113, FLOOR_CARPET: 114, FLOOR_STONE: 115,
    COUNTER: 116, TABLE: 117, CHAIR_L: 118, CHAIR_R: 119,
    BED_T: 120, BED_B: 121, BOOKSHELF: 122, PLANT_POT: 123,
    LAMP_FLOOR: 124, RUG: 125, STAIRS_U: 126, STAIRS_D: 127
};

// Building definitions with tile positions and entrance points
const BUILDINGS = {
    "Lin Family Home": {
        x: 3, y: 3, width: 6, height: 5,
        entrance: { x: 5, y: 8 },
        wallType: 'brown', label: "Lin Home"
    },
    "Moreno Family Home": {
        x: 12, y: 3, width: 6, height: 5,
        entrance: { x: 14, y: 8 },
        wallType: 'brown', label: "Moreno Home"
    },
    "Moore Family Home": {
        x: 21, y: 3, width: 6, height: 5,
        entrance: { x: 23, y: 8 },
        wallType: 'brown', label: "Moore Home"
    },
    "The Willows": {
        x: 30, y: 3, width: 8, height: 6,
        entrance: { x: 33, y: 9 },
        wallType: 'brown', label: "The Willows"
    },
    "Library": {
        x: 3, y: 14, width: 7, height: 5,
        entrance: { x: 6, y: 19 },
        wallType: 'gray', label: "Library"
    },
    "Town Hall": {
        x: 13, y: 14, width: 8, height: 6,
        entrance: { x: 16, y: 20 },
        wallType: 'gray', label: "Town Hall"
    },
    "Pharmacy": {
        x: 24, y: 14, width: 5, height: 5,
        entrance: { x: 26, y: 19 },
        wallType: 'gray', label: "Pharmacy"
    },
    "Harvey Oak Supply Store": {
        x: 32, y: 14, width: 7, height: 5,
        entrance: { x: 35, y: 19 },
        wallType: 'blue', label: "Supply Store"
    },
    "The Rose and Crown Pub": {
        x: 3, y: 26, width: 7, height: 5,
        entrance: { x: 6, y: 31 },
        wallType: 'red', label: "Pub"
    },
    "Hobbs Cafe": {
        x: 13, y: 26, width: 6, height: 5,
        entrance: { x: 15, y: 31 },
        wallType: 'red', label: "Cafe"
    },
    "Johnson Park": {
        x: 22, y: 25, width: 10, height: 8,
        entrance: { x: 27, y: 33 },
        wallType: 'park', label: "Johnson Park"
    },
    "Oak Hill College": {
        x: 44, y: 14, width: 12, height: 8,
        entrance: { x: 49, y: 22 },
        wallType: 'blue', label: "Oak Hill College"
    },
    "Williams Residence": {
        x: 44, y: 3, width: 4, height: 4,
        entrance: { x: 45, y: 7 },
        wallType: 'brown', label: "Williams"
    },
    "Anderson Residence": {
        x: 44, y: 25, width: 5, height: 4,
        entrance: { x: 46, y: 29 },
        wallType: 'brown', label: "Anderson"
    },
    "Davis Residence": {
        x: 51, y: 25, width: 5, height: 4,
        entrance: { x: 53, y: 29 },
        wallType: 'brown', label: "Davis"
    },
    "Mayor Residence": {
        x: 40, y: 3, width: 4, height: 4,
        entrance: { x: 41, y: 7 },
        wallType: 'gray', label: "Mayor"
    },
    "Rodriguez Home": {
        x: 3, y: 33, width: 5, height: 4,
        entrance: { x: 5, y: 37 },
        wallType: 'brown', label: "Rodriguez"
    },
    "Peterson Cottage": {
        x: 12, y: 33, width: 4, height: 4,
        entrance: { x: 13, y: 37 },
        wallType: 'brown', label: "Peterson"
    },
    "Thompson Residence": {
        x: 35, y: 33, width: 5, height: 4,
        entrance: { x: 37, y: 37 },
        wallType: 'brown', label: "Thompson"
    },
    "Wilson Apartment": {
        x: 42, y: 33, width: 4, height: 4,
        entrance: { x: 43, y: 37 },
        wallType: 'brown', label: "Wilson"
    }
};

// ============================================================================
// PHASER GAME CONFIGURATION
// ============================================================================

let phaserGame = null;
let mainScene = null;

// ============================================================================
// MAIN SCENE
// ============================================================================

class SmallvilleScene extends Phaser.Scene {
    constructor() {
        super({ key: 'SmallvilleScene' });
        this.agents = {};
        this.agentSprites = {};
        this.agentBubbles = {};
        this.agentLabels = {};
        this.buildingLabels = {};
        this.highlightedAgent = null;
        this.walkableGrid = [];
        this.dayNightOverlay = null;
        this.currentHour = 8;
        this.conversationIndicators = {};
    }

    preload() {
        // Load tileset
        this.load.image('tileset', '/static/assets/tileset.png');
        // Load character spritesheet
        this.load.spritesheet('character', '/static/assets/characters.png', {
            frameWidth: 16,
            frameHeight: 16
        });
    }

    create() {
        // Store reference
        mainScene = this;

        // Generate tilemap data
        this.generateTilemap();

        // Create tilemap
        this.createTilemap();

        // Create building labels
        this.createBuildingLabels();

        // Create agent animations
        this.createAnimations();

        // Setup camera controls
        this.setupCamera();

        // Create day/night overlay
        this.createDayNightOverlay();

        // Setup input handlers
        this.setupInputHandlers();

        // Initialize agents at their default positions
        this.initializeAgents();
    }

    update(time, delta) {
        // Update agent movements
        this.updateAgentMovements(delta);

        // Update speech bubbles positions
        this.updateBubbles();
    }

    // ========================================================================
    // TILEMAP GENERATION
    // ========================================================================

    generateTilemap() {
        // Create base ground layer (all grass)
        this.groundData = [];
        this.buildingData = [];
        this.decorData = [];

        for (let y = 0; y < MAP_HEIGHT; y++) {
            this.groundData[y] = [];
            this.buildingData[y] = [];
            this.decorData[y] = [];
            for (let x = 0; x < MAP_WIDTH; x++) {
                // Vary grass tiles
                const grassVariant = Math.random() < 0.1 ? TILES.GRASS_DARK :
                                    Math.random() < 0.05 ? TILES.GRASS_FLOWERS : TILES.GRASS;
                this.groundData[y][x] = grassVariant;
                this.buildingData[y][x] = -1;
                this.decorData[y][x] = -1;
            }
        }

        // Create walkable grid for pathfinding
        this.walkableGrid = [];
        for (let y = 0; y < MAP_HEIGHT; y++) {
            this.walkableGrid[y] = [];
            for (let x = 0; x < MAP_WIDTH; x++) {
                this.walkableGrid[y][x] = true; // Default walkable
            }
        }

        // Draw roads
        this.drawRoads();

        // Draw buildings
        this.drawBuildings();

        // Draw park
        this.drawPark();

        // Add decorations (trees, bushes)
        this.addDecorations();
    }

    drawRoads() {
        // Main horizontal roads
        const roadRows = [10, 11, 22, 23, 34, 35];

        for (const row of roadRows) {
            if (row >= MAP_HEIGHT) continue;
            for (let x = 0; x < MAP_WIDTH; x++) {
                this.groundData[row][x] = TILES.ROAD_H;
            }
        }

        // Main vertical roads
        const roadCols = [10, 11, 28, 29, 42, 43];

        for (const col of roadCols) {
            if (col >= MAP_WIDTH) continue;
            for (let y = 0; y < MAP_HEIGHT; y++) {
                // Handle intersections
                if (roadRows.includes(y)) {
                    this.groundData[y][col] = TILES.ROAD_CROSS;
                } else {
                    this.groundData[y][col] = TILES.ROAD_V;
                }
            }
        }
    }

    drawBuildings() {
        for (const [name, building] of Object.entries(BUILDINGS)) {
            if (building.wallType === 'park') continue; // Park handled separately

            const { x, y, width, height, wallType, entrance } = building;

            // Get wall tiles based on type
            let wallTiles;
            switch (wallType) {
                case 'brown':
                    wallTiles = {
                        tl: TILES.WALL_BROWN_TL, t: TILES.WALL_BROWN_T, tr: TILES.WALL_BROWN_TR,
                        l: TILES.WALL_BROWN_L, c: TILES.WALL_BROWN_C, r: TILES.WALL_BROWN_R,
                        bl: TILES.WALL_BROWN_BL, b: TILES.WALL_BROWN_B, br: TILES.WALL_BROWN_BR,
                        door: TILES.DOOR_BROWN, window: TILES.WINDOW_BROWN,
                        roofL: TILES.ROOF_BROWN_L, roofC: TILES.ROOF_BROWN_C, roofR: TILES.ROOF_BROWN_R
                    };
                    break;
                case 'gray':
                    wallTiles = {
                        tl: TILES.WALL_GRAY_TL, t: TILES.WALL_GRAY_T, tr: TILES.WALL_GRAY_TR,
                        l: TILES.WALL_GRAY_L, c: TILES.WALL_GRAY_C, r: TILES.WALL_GRAY_R,
                        bl: TILES.WALL_GRAY_BL, b: TILES.WALL_GRAY_B, br: TILES.WALL_GRAY_BR,
                        door: TILES.DOOR_GRAY, window: TILES.WINDOW_GRAY,
                        roofL: TILES.ROOF_GRAY_L, roofC: TILES.ROOF_GRAY_C, roofR: TILES.ROOF_GRAY_R
                    };
                    break;
                case 'red':
                    wallTiles = {
                        tl: TILES.WALL_RED_TL, t: TILES.WALL_RED_T, tr: TILES.WALL_RED_TR,
                        l: TILES.WALL_RED_L, c: TILES.WALL_RED_C, r: TILES.WALL_RED_R,
                        bl: TILES.WALL_RED_BL, b: TILES.WALL_RED_B, br: TILES.WALL_RED_BR,
                        door: TILES.DOOR_RED, window: TILES.WINDOW_RED,
                        roofL: TILES.ROOF_RED_L, roofC: TILES.ROOF_RED_C, roofR: TILES.ROOF_RED_R
                    };
                    break;
                case 'blue':
                    wallTiles = {
                        tl: TILES.WALL_BLUE_TL, t: TILES.WALL_BLUE_T, tr: TILES.WALL_BLUE_TR,
                        l: TILES.WALL_BLUE_L, c: TILES.WALL_BLUE_C, r: TILES.WALL_BLUE_R,
                        bl: TILES.WALL_BLUE_BL, b: TILES.WALL_BLUE_B, br: TILES.WALL_BLUE_BR,
                        door: TILES.DOOR_BLUE, window: TILES.WINDOW_BLUE,
                        roofL: TILES.ROOF_BLUE_L, roofC: TILES.ROOF_BLUE_C, roofR: TILES.ROOF_BLUE_R
                    };
                    break;
            }

            // Draw roof (1 row above building)
            const roofY = y - 1;
            if (roofY >= 0) {
                this.buildingData[roofY][x] = wallTiles.roofL;
                for (let rx = x + 1; rx < x + width - 1; rx++) {
                    if (rx < MAP_WIDTH) this.buildingData[roofY][rx] = wallTiles.roofC;
                }
                if (x + width - 1 < MAP_WIDTH) this.buildingData[roofY][x + width - 1] = wallTiles.roofR;
            }

            // Draw building walls
            for (let by = y; by < y + height && by < MAP_HEIGHT; by++) {
                for (let bx = x; bx < x + width && bx < MAP_WIDTH; bx++) {
                    let tile;
                    const isTop = by === y;
                    const isBottom = by === y + height - 1;
                    const isLeft = bx === x;
                    const isRight = bx === x + width - 1;

                    // Check if this is the door position
                    const isDoor = bx === entrance.x && by === y + height - 1;

                    if (isDoor) {
                        tile = wallTiles.door;
                    } else if (isTop && isLeft) {
                        tile = wallTiles.tl;
                    } else if (isTop && isRight) {
                        tile = wallTiles.tr;
                    } else if (isBottom && isLeft) {
                        tile = wallTiles.bl;
                    } else if (isBottom && isRight) {
                        tile = wallTiles.br;
                    } else if (isTop) {
                        tile = wallTiles.t;
                    } else if (isBottom) {
                        tile = wallTiles.b;
                    } else if (isLeft) {
                        tile = wallTiles.l;
                    } else if (isRight) {
                        tile = wallTiles.r;
                    } else {
                        // Interior - add windows sometimes
                        tile = (bx + by) % 3 === 0 ? wallTiles.window : wallTiles.c;
                    }

                    this.buildingData[by][bx] = tile;
                    // Buildings are not walkable (except doors are semi-walkable for pathfinding)
                    if (!isDoor) {
                        this.walkableGrid[by][bx] = false;
                    }
                }
            }
        }
    }

    drawPark() {
        const park = BUILDINGS["Johnson Park"];
        const { x, y, width, height } = park;

        // Park is special - it's walkable green space
        for (let py = y; py < y + height && py < MAP_HEIGHT; py++) {
            for (let px = x; px < x + width && px < MAP_WIDTH; px++) {
                this.groundData[py][px] = TILES.PARK_GRASS;
            }
        }

        // Add pond in center
        const pondX = x + 3;
        const pondY = y + 2;
        this.decorData[pondY][pondX] = TILES.POND_TL;
        this.decorData[pondY][pondX + 1] = TILES.POND_TR;
        this.decorData[pondY + 1][pondX] = TILES.POND_BL;
        this.decorData[pondY + 1][pondX + 1] = TILES.POND_BR;
        // Pond not walkable
        this.walkableGrid[pondY][pondX] = false;
        this.walkableGrid[pondY][pondX + 1] = false;
        this.walkableGrid[pondY + 1][pondX] = false;
        this.walkableGrid[pondY + 1][pondX + 1] = false;

        // Add benches
        this.decorData[y + 1][x + 1] = TILES.PARK_BENCH;
        this.decorData[y + height - 2][x + width - 2] = TILES.PARK_BENCH;

        // Add lamp
        this.decorData[y + height - 2][x + 1] = TILES.PARK_LAMP;

        // Add playground
        this.decorData[y + 4][x + 6] = TILES.PLAYGROUND;
    }

    addDecorations() {
        // Add trees around the map edges
        const treePositions = [
            [1, 1], [1, 12], [1, 25], [1, 37],
            [57, 1], [57, 12], [57, 25], [57, 37],
            [20, 1], [40, 1],
            [20, 37], [50, 37]
        ];

        for (const [tx, ty] of treePositions) {
            if (tx < MAP_WIDTH && ty < MAP_HEIGHT && ty > 0) {
                this.decorData[ty - 1][tx] = TILES.TREE_TOP;
                this.decorData[ty][tx] = TILES.TREE_BOTTOM;
                this.walkableGrid[ty - 1][tx] = false;
                this.walkableGrid[ty][tx] = false;
            }
        }

        // Add bushes near buildings
        for (const building of Object.values(BUILDINGS)) {
            if (building.wallType === 'park') continue;
            // Add bush near entrance
            const bushX = building.entrance.x + 2;
            const bushY = building.entrance.y;
            if (bushX < MAP_WIDTH && bushY < MAP_HEIGHT &&
                this.groundData[bushY][bushX] === TILES.GRASS) {
                this.decorData[bushY][bushX] = TILES.BUSH;
            }
        }

        // Add flower patches
        const flowerPositions = [
            [5, 12, TILES.FLOWERS_RED],
            [15, 12, TILES.FLOWERS_YELLOW],
            [25, 12, TILES.FLOWERS_BLUE],
            [35, 24, TILES.FLOWERS_RED]
        ];

        for (const [fx, fy, tile] of flowerPositions) {
            if (fx < MAP_WIDTH && fy < MAP_HEIGHT &&
                this.groundData[fy][fx] === TILES.GRASS) {
                this.decorData[fy][fx] = tile;
            }
        }
    }

    createTilemap() {
        // Create tilemap from data
        const mapData = {
            width: MAP_WIDTH,
            height: MAP_HEIGHT,
            tilewidth: TILE_SIZE,
            tileheight: TILE_SIZE
        };

        this.map = this.make.tilemap({
            tileWidth: TILE_SIZE,
            tileHeight: TILE_SIZE,
            width: MAP_WIDTH,
            height: MAP_HEIGHT
        });

        // Add tileset image
        const tileset = this.map.addTilesetImage('tileset', 'tileset', TILE_SIZE, TILE_SIZE, 0, 0);

        // Create ground layer
        this.groundLayer = this.map.createBlankLayer('ground', tileset);
        for (let y = 0; y < MAP_HEIGHT; y++) {
            for (let x = 0; x < MAP_WIDTH; x++) {
                this.groundLayer.putTileAt(this.groundData[y][x], x, y);
            }
        }

        // Create building layer
        this.buildingLayer = this.map.createBlankLayer('buildings', tileset);
        for (let y = 0; y < MAP_HEIGHT; y++) {
            for (let x = 0; x < MAP_WIDTH; x++) {
                if (this.buildingData[y][x] >= 0) {
                    this.buildingLayer.putTileAt(this.buildingData[y][x], x, y);
                }
            }
        }

        // Create decoration layer
        this.decorLayer = this.map.createBlankLayer('decor', tileset);
        for (let y = 0; y < MAP_HEIGHT; y++) {
            for (let x = 0; x < MAP_WIDTH; x++) {
                if (this.decorData[y][x] >= 0) {
                    this.decorLayer.putTileAt(this.decorData[y][x], x, y);
                }
            }
        }

        // Set world bounds
        this.physics.world.setBounds(0, 0, MAP_WIDTH * TILE_SIZE, MAP_HEIGHT * TILE_SIZE);
    }

    createBuildingLabels() {
        for (const [name, building] of Object.entries(BUILDINGS)) {
            const labelX = (building.x + building.width / 2) * TILE_SIZE;
            const labelY = (building.y - 1.5) * TILE_SIZE;

            const label = this.add.text(labelX, labelY, building.label, {
                fontSize: '10px',
                fontFamily: 'Arial',
                color: '#ffffff',
                stroke: '#000000',
                strokeThickness: 3,
                align: 'center'
            }).setOrigin(0.5, 0.5).setDepth(100);

            this.buildingLabels[name] = label;
        }
    }

    // ========================================================================
    // ANIMATIONS
    // ========================================================================

    createAnimations() {
        // Walk down
        this.anims.create({
            key: 'walk_down',
            frames: this.anims.generateFrameNumbers('character', { frames: [0, 1] }),
            frameRate: 6,
            repeat: -1
        });

        // Walk left
        this.anims.create({
            key: 'walk_left',
            frames: this.anims.generateFrameNumbers('character', { frames: [2, 3] }),
            frameRate: 6,
            repeat: -1
        });

        // Walk right
        this.anims.create({
            key: 'walk_right',
            frames: this.anims.generateFrameNumbers('character', { frames: [4, 5] }),
            frameRate: 6,
            repeat: -1
        });

        // Walk up
        this.anims.create({
            key: 'walk_up',
            frames: this.anims.generateFrameNumbers('character', { frames: [6, 7] }),
            frameRate: 6,
            repeat: -1
        });

        // Idle (front facing)
        this.anims.create({
            key: 'idle',
            frames: [{ key: 'character', frame: 0 }],
            frameRate: 1,
            repeat: 0
        });
    }

    // ========================================================================
    // CAMERA CONTROLS
    // ========================================================================

    setupCamera() {
        const camera = this.cameras.main;

        // Set camera bounds
        camera.setBounds(0, 0, MAP_WIDTH * TILE_SIZE, MAP_HEIGHT * TILE_SIZE);

        // Center camera
        camera.centerOn(MAP_WIDTH * TILE_SIZE / 2, MAP_HEIGHT * TILE_SIZE / 2);

        // Enable zoom
        camera.setZoom(2);

        // Enable second pointer for multitouch
        this.input.addPointer(1);

        // Track pinch distance for pinch-to-zoom
        let pinchDistance = 0;
        let isPinching = false;

        // Mouse drag to pan (single touch or mouse)
        this.input.on('pointermove', (pointer) => {
            // Skip if pinching
            if (isPinching) return;

            if (pointer.isDown && pointer.button === 0) {
                // Only pan with single touch
                if (!this.input.pointer2.isDown) {
                    camera.scrollX -= (pointer.x - pointer.prevPosition.x) / camera.zoom;
                    camera.scrollY -= (pointer.y - pointer.prevPosition.y) / camera.zoom;
                }
            }
        });

        // Pinch-to-zoom for touch devices
        this.input.on('pointerdown', (pointer) => {
            if (this.input.pointer1.isDown && this.input.pointer2.isDown) {
                const dx = this.input.pointer1.x - this.input.pointer2.x;
                const dy = this.input.pointer1.y - this.input.pointer2.y;
                pinchDistance = Math.sqrt(dx * dx + dy * dy);
                isPinching = true;
            }
        });

        this.input.on('pointermove', (pointer) => {
            if (this.input.pointer1.isDown && this.input.pointer2.isDown) {
                const dx = this.input.pointer1.x - this.input.pointer2.x;
                const dy = this.input.pointer1.y - this.input.pointer2.y;
                const newDistance = Math.sqrt(dx * dx + dy * dy);

                if (pinchDistance > 0) {
                    const zoomDelta = (newDistance - pinchDistance) * 0.005;
                    const newZoom = Phaser.Math.Clamp(camera.zoom + zoomDelta, 0.5, 4);
                    camera.setZoom(newZoom);
                    pinchDistance = newDistance;
                }
            }
        });

        this.input.on('pointerup', (pointer) => {
            // Reset pinch state when any pointer is released
            if (!this.input.pointer1.isDown || !this.input.pointer2.isDown) {
                isPinching = false;
                pinchDistance = 0;
            }
        });

        // Mouse wheel to zoom
        this.input.on('wheel', (pointer, gameObjects, deltaX, deltaY, deltaZ) => {
            const zoomChange = deltaY > 0 ? -0.1 : 0.1;
            const newZoom = Phaser.Math.Clamp(camera.zoom + zoomChange, 0.5, 4);
            camera.setZoom(newZoom);
        });
    }

    // ========================================================================
    // DAY/NIGHT CYCLE
    // ========================================================================

    createDayNightOverlay() {
        // Create a rectangle that covers the entire map
        this.dayNightOverlay = this.add.rectangle(
            MAP_WIDTH * TILE_SIZE / 2,
            MAP_HEIGHT * TILE_SIZE / 2,
            MAP_WIDTH * TILE_SIZE,
            MAP_HEIGHT * TILE_SIZE,
            0x000033,
            0
        ).setDepth(500);
    }

    updateDayNight(hour) {
        this.currentHour = hour;

        // Calculate alpha based on time of day
        let alpha = 0;

        if (hour >= 6 && hour < 8) {
            // Dawn: gradually decrease darkness
            alpha = 0.3 * (1 - (hour - 6) / 2);
        } else if (hour >= 8 && hour < 18) {
            // Day: no overlay
            alpha = 0;
        } else if (hour >= 18 && hour < 20) {
            // Dusk: gradually increase darkness
            alpha = 0.3 * ((hour - 18) / 2);
        } else if (hour >= 20 || hour < 6) {
            // Night: maximum darkness
            alpha = 0.4;
        }

        if (this.dayNightOverlay) {
            this.dayNightOverlay.setAlpha(alpha);
        }
    }

    // ========================================================================
    // INPUT HANDLERS
    // ========================================================================

    setupInputHandlers() {
        // Click to select agent
        this.input.on('pointerup', (pointer) => {
            // Ignore if we were dragging
            if (Math.abs(pointer.x - pointer.downX) > 5 || Math.abs(pointer.y - pointer.downY) > 5) {
                return;
            }

            const worldPoint = this.cameras.main.getWorldPoint(pointer.x, pointer.y);

            // Check if clicked on an agent
            for (const [agentName, sprite] of Object.entries(this.agentSprites)) {
                const bounds = sprite.getBounds();
                if (bounds.contains(worldPoint.x, worldPoint.y)) {
                    this.selectAgent(agentName);
                    return;
                }
            }
        });
    }

    selectAgent(agentName) {
        // Clear previous highlight and hide its bubble
        if (this.highlightedAgent && this.agentSprites[this.highlightedAgent]) {
            this.agentSprites[this.highlightedAgent].clearTint();
            // Remove highlight circle
            if (this.highlightCircle) {
                this.highlightCircle.destroy();
                this.highlightCircle = null;
            }
            // Hide previous agent's bubble
            this.updateSpeechBubble(this.highlightedAgent, null);
        }

        this.highlightedAgent = agentName;

        // Add highlight to new agent
        if (this.agentSprites[agentName]) {
            const sprite = this.agentSprites[agentName];

            // Create highlight circle
            this.highlightCircle = this.add.circle(sprite.x, sprite.y, 12)
                .setStrokeStyle(2, 0xffffff)
                .setDepth(199);
        }

        // Notify app.js
        if (window.onAgentClick) {
            window.onAgentClick(agentName);
        }
    }

    // ========================================================================
    // AGENT MANAGEMENT
    // ========================================================================

    initializeAgents() {
        // Create sprites for all known agents at default positions
        AGENT_NAMES.forEach((name, index) => {
            const color = Phaser.Display.Color.HSLToColor(index / 25, 0.7, 0.55).color;
            this.createAgentSprite(name, color);
        });
    }

    createAgentSprite(agentName, tintColor) {
        // Default position (will be updated when data arrives)
        const startX = MAP_WIDTH * TILE_SIZE / 2;
        const startY = MAP_HEIGHT * TILE_SIZE / 2;

        const sprite = this.add.sprite(startX, startY, 'character')
            .setDepth(200)
            .setTint(tintColor)
            .setInteractive();

        sprite.agentName = agentName;
        sprite.targetX = startX;
        sprite.targetY = startY;
        sprite.path = [];
        sprite.pathIndex = 0;
        sprite.isMoving = false;
        sprite.tintColor = tintColor;

        // Start with idle animation
        sprite.play('idle');

        this.agentSprites[agentName] = sprite;

        // Create speech bubble (hidden initially)
        const bubble = this.createSpeechBubble(agentName);
        this.agentBubbles[agentName] = bubble;

        // Create name label below sprite
        const firstName = agentName.split(' ')[0];
        const label = this.add.text(startX, startY + 12, firstName, {
            fontSize: '7px',
            fontFamily: 'Arial',
            color: '#ffffff',
            stroke: '#000000',
            strokeThickness: 2,
            align: 'center'
        }).setOrigin(0.5, 0).setDepth(201);
        this.agentLabels[agentName] = label;
    }

    createSpeechBubble(agentName) {
        const container = this.add.container(0, 0).setDepth(300).setVisible(false);

        // Background
        const bg = this.add.graphics();
        bg.fillStyle(0xffffff, 0.9);
        bg.fillRoundedRect(-60, -30, 120, 24, 6);
        bg.lineStyle(1, 0x333333, 1);
        bg.strokeRoundedRect(-60, -30, 120, 24, 6);
        // Triangle pointer
        bg.fillStyle(0xffffff, 0.9);
        bg.fillTriangle(-5, -6, 5, -6, 0, 2);

        container.add(bg);

        // Text
        const text = this.add.text(0, -18, '', {
            fontSize: '8px',
            fontFamily: 'Arial',
            color: '#333333',
            align: 'center',
            wordWrap: { width: 110 }
        }).setOrigin(0.5, 0.5);

        container.add(text);
        container.textObj = text;
        container.bgObj = bg;

        return container;
    }

    updateSpeechBubble(agentName, activity) {
        const bubble = this.agentBubbles[agentName];
        if (!bubble) return;

        if (activity && activity.length > 0) {
            // Truncate to ~30 chars
            let text = activity.length > 30 ? activity.substring(0, 28) + '...' : activity;
            bubble.textObj.setText(text);
            bubble.setVisible(true);
        } else {
            bubble.setVisible(false);
        }
    }

    updateBubbles() {
        for (const [agentName, sprite] of Object.entries(this.agentSprites)) {
            const bubble = this.agentBubbles[agentName];
            if (bubble && sprite) {
                bubble.setPosition(sprite.x, sprite.y - 20);
            }
            // Update name label position
            const label = this.agentLabels[agentName];
            if (label && sprite) {
                label.setPosition(sprite.x, sprite.y + 12);
            }
        }

        // Update highlight circle position
        if (this.highlightCircle && this.highlightedAgent && this.agentSprites[this.highlightedAgent]) {
            const sprite = this.agentSprites[this.highlightedAgent];
            this.highlightCircle.setPosition(sprite.x, sprite.y);
        }

        // Update conversation indicators
        for (const [key, indicator] of Object.entries(this.conversationIndicators)) {
            if (indicator.agent1 && indicator.agent2) {
                const sprite1 = this.agentSprites[indicator.agent1];
                const sprite2 = this.agentSprites[indicator.agent2];
                if (sprite1 && sprite2) {
                    indicator.setPosition(
                        (sprite1.x + sprite2.x) / 2,
                        (sprite1.y + sprite2.y) / 2 - 16
                    );
                }
            }
        }
    }

    // ========================================================================
    // A* PATHFINDING
    // ========================================================================

    findPath(startX, startY, endX, endY) {
        // Convert world coords to tile coords
        const startTileX = Math.floor(startX / TILE_SIZE);
        const startTileY = Math.floor(startY / TILE_SIZE);
        const endTileX = Math.floor(endX / TILE_SIZE);
        const endTileY = Math.floor(endY / TILE_SIZE);

        // Clamp to map bounds
        const clampedEnd = {
            x: Phaser.Math.Clamp(endTileX, 0, MAP_WIDTH - 1),
            y: Phaser.Math.Clamp(endTileY, 0, MAP_HEIGHT - 1)
        };

        // Simple A* implementation
        const openSet = [];
        const closedSet = new Set();
        const cameFrom = {};

        const start = { x: startTileX, y: startTileY, g: 0, h: 0, f: 0 };
        start.h = this.heuristic(start, clampedEnd);
        start.f = start.h;
        openSet.push(start);

        const getKey = (node) => `${node.x},${node.y}`;

        while (openSet.length > 0) {
            // Get node with lowest f score
            openSet.sort((a, b) => a.f - b.f);
            const current = openSet.shift();

            // Check if we reached the goal
            if (current.x === clampedEnd.x && current.y === clampedEnd.y) {
                // Reconstruct path
                const path = [];
                let node = current;
                while (node) {
                    path.unshift({
                        x: node.x * TILE_SIZE + TILE_SIZE / 2,
                        y: node.y * TILE_SIZE + TILE_SIZE / 2
                    });
                    node = cameFrom[getKey(node)];
                }
                return path;
            }

            closedSet.add(getKey(current));

            // Check neighbors (4-directional)
            const neighbors = [
                { x: current.x + 1, y: current.y },
                { x: current.x - 1, y: current.y },
                { x: current.x, y: current.y + 1 },
                { x: current.x, y: current.y - 1 }
            ];

            for (const neighbor of neighbors) {
                // Check bounds
                if (neighbor.x < 0 || neighbor.x >= MAP_WIDTH ||
                    neighbor.y < 0 || neighbor.y >= MAP_HEIGHT) {
                    continue;
                }

                // Check if walkable
                if (!this.walkableGrid[neighbor.y][neighbor.x]) {
                    continue;
                }

                const key = getKey(neighbor);
                if (closedSet.has(key)) {
                    continue;
                }

                const g = current.g + 1;
                const h = this.heuristic(neighbor, clampedEnd);
                const f = g + h;

                // Check if already in open set with better score
                const existing = openSet.find(n => n.x === neighbor.x && n.y === neighbor.y);
                if (existing && existing.g <= g) {
                    continue;
                }

                if (existing) {
                    existing.g = g;
                    existing.f = f;
                    cameFrom[key] = current;
                } else {
                    openSet.push({ x: neighbor.x, y: neighbor.y, g, h, f });
                    cameFrom[key] = current;
                }
            }
        }

        // No path found - return direct line
        return [
            { x: startX, y: startY },
            { x: clampedEnd.x * TILE_SIZE + TILE_SIZE / 2, y: clampedEnd.y * TILE_SIZE + TILE_SIZE / 2 }
        ];
    }

    heuristic(a, b) {
        // Manhattan distance
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
    }

    // ========================================================================
    // AGENT MOVEMENT
    // ========================================================================

    resolveBuilding(location) {
        let building = BUILDINGS[location];
        if (building) return building;

        // Try fuzzy matching for hallucinated/non-standard location names
        const locationLower = location.toLowerCase();
        const LOCATION_HINTS = {
            'restaurant': 'The Rose and Crown Pub',
            'diner': 'The Rose and Crown Pub',
            'clinic': 'Pharmacy',
            'medical': 'Pharmacy',
            'doctor': 'Pharmacy',
            'hospital': 'Pharmacy',
            'police': 'Town Hall',
            'station': 'Town Hall',
            'campus': 'Oak Hill College',
            'school': 'Oak Hill College',
            'college': 'Oak Hill College',
            'store': 'Harvey Oak Supply Store',
            'shop': 'Harvey Oak Supply Store',
            'supply': 'Harvey Oak Supply Store',
            'garden': 'Johnson Park',
            'park': 'Johnson Park',
            'pub': 'The Rose and Crown Pub',
            'bar': 'The Rose and Crown Pub',
            'cafe': 'Hobbs Cafe',
            'coffee': 'Hobbs Cafe',
            'library': 'Library',
            'book': 'Library',
            'town hall': 'Town Hall',
            'mayor': 'Town Hall',
            'pharmacy': 'Pharmacy',
            'willows': 'The Willows',
            'apartment': 'The Willows',
        };
        for (const [hint, target] of Object.entries(LOCATION_HINTS)) {
            if (locationLower.includes(hint)) {
                building = BUILDINGS[target];
                if (building) return building;
            }
        }
        // Last resort: substring match
        for (const [bName, bData] of Object.entries(BUILDINGS)) {
            if (bName.toLowerCase().includes(locationLower) || locationLower.includes(bName.toLowerCase())) {
                return bData;
            }
        }
        return null;
    }

    moveAgentTo(agentName, location, offset = { dx: 0, dy: 0 }) {
        const sprite = this.agentSprites[agentName];
        if (!sprite) return;

        const building = this.resolveBuilding(location);
        if (!building) {
            console.warn(`[map] No building found for location: "${location}" (agent: ${agentName})`);
            return;
        }

        const targetX = building.entrance.x * TILE_SIZE + TILE_SIZE / 2 + offset.dx;
        const targetY = building.entrance.y * TILE_SIZE + TILE_SIZE / 2 + offset.dy;

        // Find path to location
        const path = this.findPath(sprite.x, sprite.y, targetX, targetY);

        sprite.path = path;
        sprite.pathIndex = 0;
        sprite.isMoving = path.length > 1;

        if (sprite.isMoving) {
            this.updateAgentAnimation(sprite);
        }
    }

    /**
     * Gently nudge an agent to a new position within the same location cluster.
     * Smooth direct movement (no pathfinding needed — they're already at the building).
     */
    nudgeAgentTo(sprite, location, offset) {
        const building = this.resolveBuilding(location);
        if (!building) return;

        const targetX = building.entrance.x * TILE_SIZE + TILE_SIZE / 2 + offset.dx;
        const targetY = building.entrance.y * TILE_SIZE + TILE_SIZE / 2 + offset.dy;

        // Short direct path — just slide over
        sprite.path = [{ x: targetX, y: targetY }];
        sprite.pathIndex = 0;
        sprite.isMoving = true;
    }

    updateAgentMovements(delta) {
        const speed = 0.05; // Tiles per millisecond

        for (const [agentName, sprite] of Object.entries(this.agentSprites)) {
            if (!sprite.isMoving || !sprite.path || sprite.path.length === 0) {
                continue;
            }

            const target = sprite.path[sprite.pathIndex];
            if (!target) {
                sprite.isMoving = false;
                sprite.play('idle');
                continue;
            }

            const dx = target.x - sprite.x;
            const dy = target.y - sprite.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < 2) {
                // Reached waypoint
                sprite.pathIndex++;
                if (sprite.pathIndex >= sprite.path.length) {
                    sprite.isMoving = false;
                    sprite.play('idle');
                } else {
                    this.updateAgentAnimation(sprite);
                }
            } else {
                // Move towards target
                const moveX = (dx / distance) * speed * delta;
                const moveY = (dy / distance) * speed * delta;

                sprite.x += moveX;
                sprite.y += moveY;
            }
        }
    }

    updateAgentAnimation(sprite) {
        if (!sprite.path || sprite.pathIndex >= sprite.path.length) return;

        const target = sprite.path[sprite.pathIndex];
        const dx = target.x - sprite.x;
        const dy = target.y - sprite.y;

        // Determine direction
        if (Math.abs(dx) > Math.abs(dy)) {
            sprite.play(dx > 0 ? 'walk_right' : 'walk_left', true);
        } else {
            sprite.play(dy > 0 ? 'walk_down' : 'walk_up', true);
        }
    }

    // ========================================================================
    // CONVERSATION INDICATORS
    // ========================================================================

    showConversationIndicator(agent1, agent2) {
        const key = [agent1, agent2].sort().join('|');

        if (this.conversationIndicators[key]) {
            return; // Already showing
        }

        // Create chat icon
        const indicator = this.add.text(0, 0, '💬', {
            fontSize: '12px'
        }).setOrigin(0.5, 0.5).setDepth(400);

        indicator.agent1 = agent1;
        indicator.agent2 = agent2;

        this.conversationIndicators[key] = indicator;
    }

    hideConversationIndicator(agent1, agent2) {
        const key = [agent1, agent2].sort().join('|');

        if (this.conversationIndicators[key]) {
            this.conversationIndicators[key].destroy();
            delete this.conversationIndicators[key];
        }
    }

    clearAllConversationIndicators() {
        for (const key of Object.keys(this.conversationIndicators)) {
            this.conversationIndicators[key].destroy();
        }
        this.conversationIndicators = {};
    }

    // ========================================================================
    // AGENT CLUSTERING (fan-out around entrance)
    // ========================================================================

    /**
     * Calculate offset position for an agent within a group at the same location.
     * Arranges agents in a semicircle arc below the entrance point.
     * @param {number} index - Agent's index within the location group (0-based)
     * @param {number} total - Total agents at this location
     * @returns {{dx: number, dy: number}} Pixel offset from entrance
     */
    getClusterOffset(index, total) {
        if (total <= 1) return { dx: 0, dy: 0 };

        // Ring configuration: inner ring holds up to 6, overflow goes to outer ring
        const INNER_RADIUS = 14;
        const OUTER_RADIUS = 26;
        const INNER_MAX = 6;

        let ring, ringIndex, ringTotal, radius;

        if (total <= INNER_MAX) {
            // All fit in inner ring
            ring = 0;
            ringIndex = index;
            ringTotal = total;
            radius = INNER_RADIUS;
        } else if (index < INNER_MAX) {
            // Inner ring
            ring = 0;
            ringIndex = index;
            ringTotal = INNER_MAX;
            radius = INNER_RADIUS;
        } else {
            // Outer ring
            ring = 1;
            ringIndex = index - INNER_MAX;
            ringTotal = total - INNER_MAX;
            radius = OUTER_RADIUS;
        }

        // Spread across a 180° arc below the entrance (π/6 to 5π/6 to avoid stacking directly below)
        const arcStart = Math.PI / 6;
        const arcEnd = 5 * Math.PI / 6;
        const angle = ringTotal === 1
            ? Math.PI / 2  // Single agent goes straight down
            : arcStart + (arcEnd - arcStart) * (ringIndex / (ringTotal - 1));

        const dx = Math.cos(angle) * radius;
        const dy = Math.sin(angle) * radius * 0.7 + 8; // Flatten vertically, push down below entrance

        return { dx: Math.round(dx), dy: Math.round(dy) };
    }

    /**
     * Update building labels with agent count badges
     */
    updateBuildingCounts(locationGroups) {
        for (const [name, building] of Object.entries(BUILDINGS)) {
            const label = this.buildingLabels[name];
            if (!label) continue;

            const count = locationGroups[name] ? locationGroups[name].length : 0;
            if (count > 0) {
                label.setText(`${building.label} [${count}]`);
                label.setStyle({ color: count >= 5 ? '#ffdd44' : '#ffffff' });
            } else {
                label.setText(building.label);
                label.setStyle({ color: '#aaaaaa' });
            }
        }
    }

    // ========================================================================
    // EXTERNAL API (called by app.js)
    // ========================================================================

    updateFromTick(agents, locations, conversations, simTime) {
        // Update time of day
        if (simTime) {
            const date = new Date(simTime);
            this.updateDayNight(date.getHours());
        }

        // Clear old conversation indicators
        this.clearAllConversationIndicators();

        // Show active conversation indicators
        if (conversations) {
            for (const convo of conversations) {
                this.showConversationIndicator(convo.agent1, convo.agent2);
            }
        }

        // Update agent positions and activities
        if (!agents) return;

        // Group agents by location for clustering
        const locationGroups = {};
        for (const [agentName, agentData] of Object.entries(agents)) {
            const loc = agentData.location;
            if (loc) {
                if (!locationGroups[loc]) locationGroups[loc] = [];
                locationGroups[loc].push(agentName);
            }
        }
        // Sort each group alphabetically for deterministic positioning
        for (const loc of Object.keys(locationGroups)) {
            locationGroups[loc].sort();
        }

        // Update building count badges
        this.updateBuildingCounts(locationGroups);

        for (const [agentName, agentData] of Object.entries(agents)) {
            const sprite = this.agentSprites[agentName];
            if (!sprite) continue;

            // Only show speech bubble for the selected (highlighted) agent
            if (agentName === this.highlightedAgent) {
                this.updateSpeechBubble(agentName, agentData.activity);
            } else {
                this.updateSpeechBubble(agentName, null);
            }

            // Check if location changed - if so, pathfind to new location
            const currentLocation = sprite.currentLocation;
            const newLocation = agentData.location;

            if (newLocation && newLocation !== currentLocation) {
                sprite.currentLocation = newLocation;

                // Calculate cluster offset
                const group = locationGroups[newLocation] || [agentName];
                const index = group.indexOf(agentName);
                const offset = this.getClusterOffset(index >= 0 ? index : 0, group.length);

                this.moveAgentTo(agentName, newLocation, offset);
            } else if (newLocation && sprite.currentClusterTotal !== (locationGroups[newLocation] || []).length) {
                // Same location but group size changed — reposition within cluster
                const group = locationGroups[newLocation] || [agentName];
                const index = group.indexOf(agentName);
                const offset = this.getClusterOffset(index >= 0 ? index : 0, group.length);
                sprite.currentClusterTotal = group.length;

                this.nudgeAgentTo(sprite, newLocation, offset);
            }
        }
    }
}

// ============================================================================
// INITIALIZATION & EXPORTS
// ============================================================================

function initMap(containerId) {
    // Get container element
    const container = document.getElementById('phaser-container');
    if (!container) {
        console.error('Phaser container not found');
        return;
    }

    // Get container dimensions
    const rect = container.getBoundingClientRect();

    // Create Phaser game
    const config = {
        type: Phaser.AUTO,
        width: rect.width || 800,
        height: rect.height || 500,
        parent: 'phaser-container',
        pixelArt: true,
        backgroundColor: '#1a1a2e',
        scene: SmallvilleScene,
        physics: {
            default: 'arcade',
            arcade: {
                debug: false
            }
        },
        scale: {
            mode: Phaser.Scale.RESIZE,
            autoCenter: Phaser.Scale.CENTER_BOTH
        }
    };

    phaserGame = new Phaser.Game(config);

    // Handle window resize
    window.addEventListener('resize', () => {
        if (phaserGame) {
            const newRect = container.getBoundingClientRect();
            phaserGame.scale.resize(newRect.width, newRect.height);
        }
    });
}

function updateMap(agents, locations, conversations, simTime) {
    if (mainScene) {
        mainScene.updateFromTick(agents, locations, conversations, simTime);
    }
}

function highlightAgent(agentName) {
    if (mainScene) {
        mainScene.selectAgent(agentName);
    }
}

function clearHighlight() {
    if (mainScene && mainScene.highlightedAgent) {
        // Hide the bubble before clearing
        mainScene.updateSpeechBubble(mainScene.highlightedAgent, null);
        if (mainScene.highlightCircle) {
            mainScene.highlightCircle.destroy();
            mainScene.highlightCircle = null;
        }
        mainScene.highlightedAgent = null;
    }
}

// Export functions for app.js
window.initMap = initMap;
window.updateMap = updateMap;
window.highlightAgent = highlightAgent;
window.clearHighlight = clearHighlight;
window.AGENT_NAMES = AGENT_NAMES;
