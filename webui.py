"""Web UI server for Generative Agents simulation.

Provides a WebSocket-based real-time dashboard and REST API for inspecting
simulation state, agent memories, conversations, and locations.

Usage:
    webui = SmallvilleWebUI(simulation)
    await webui.start(port=8080)
    # ... in the tick loop:
    await webui.broadcast_tick()
    # ... on shutdown:
    await webui.stop()
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Set

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)


class SmallvilleWebUI:
    """Aiohttp-based web server embedded in the simulation's asyncio loop."""

    def __init__(self, simulation):
        self.simulation = simulation
        self.app = web.Application()
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self.ws_clients: set[web.WebSocketResponse] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, port: int = 8080):
        """Create the aiohttp app, register routes, and start listening."""
        self._register_routes()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", port)
        await self.site.start()
        logger.info(f"Web UI server started on http://0.0.0.0:{port}")
        
        # Start periodic broadcast so clients get updates even when tick loop is busy
        self._periodic_broadcast_task = asyncio.create_task(self._periodic_broadcast())
    
    async def _periodic_broadcast(self):
        """Broadcast state every 5 seconds regardless of tick loop progress."""
        while True:
            await asyncio.sleep(5)
            try:
                await self.broadcast_tick()
            except Exception:
                pass

    async def stop(self):
        """Gracefully close all WebSocket connections and shut down."""
        # Close every open WebSocket
        for ws in set(self.ws_clients):
            try:
                await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b"Server shutting down")
            except Exception:
                pass
        self.ws_clients.clear()

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Web UI server stopped")

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def _register_routes(self):
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/ws", self.ws_handler)

        # REST API
        self.app.router.add_get("/api/agent/{name}", self.handle_agent)
        self.app.router.add_get("/api/agent/{name}/memories", self.handle_agent_memories)
        self.app.router.add_get("/api/agent/{name}/conversations", self.handle_agent_conversations)
        self.app.router.add_get("/api/conversations", self.handle_conversations)
        self.app.router.add_get("/api/locations", self.handle_locations)
        self.app.router.add_post("/api/pause", self.handle_pause)
        self.app.router.add_post("/api/resume", self.handle_resume)

        # Static files served from webui/ directory (CSS, JS, images)
        static_dir = Path(__file__).parent / "webui"
        if static_dir.is_dir():
            self.app.router.add_static("/static", static_dir)

    # ------------------------------------------------------------------
    # Index page
    # ------------------------------------------------------------------

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serve webui/index.html at GET /."""
        index_path = Path(__file__).parent / "webui" / "index.html"
        if index_path.is_file():
            return web.FileResponse(index_path)
        # Minimal fallback if no static file exists yet
        return web.Response(
            text="<html><body><h1>Smallville Web UI</h1>"
                 "<p>Place index.html in webui/ directory.</p></body></html>",
            content_type="text/html",
        )

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Accept a WebSocket connection, relay control messages."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.add(ws)
        logger.info(f"WebSocket client connected ({len(self.ws_clients)} total)")

        # Send initial state immediately so agents aren't stuck at center
        try:
            payload = self._build_tick_state()
            await ws.send_str(json.dumps(payload))
            logger.info("Sent initial state to new WebSocket client")
        except Exception as exc:
            logger.warning(f"Failed to send initial state: {exc}")

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_ws_message(ws, msg.data)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                    break
        except Exception as exc:
            logger.warning(f"WebSocket error: {exc}")
        finally:
            self.ws_clients.discard(ws)
            logger.info(f"WebSocket client disconnected ({len(self.ws_clients)} remaining)")

        return ws

    async def _handle_ws_message(self, ws: web.WebSocketResponse, raw: str):
        """Process a control message from a WebSocket client."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "message": "Invalid JSON"})
            return

        msg_type = data.get("type", "")
        sim = self.simulation

        if msg_type == "pause":
            sim.pause(yield_vram=data.get("yield_vram", False))
            await ws.send_json({"type": "ack", "action": "pause"})

        elif msg_type == "resume":
            await sim.resume()
            await ws.send_json({"type": "ack", "action": "resume"})

        elif msg_type == "set_speed":
            speed = data.get("speed")
            if isinstance(speed, (int, float)) and speed > 0:
                sim.set_speed(int(speed))
                await ws.send_json({"type": "ack", "action": "set_speed", "speed": sim.simulation_speed})
            else:
                await ws.send_json({"type": "error", "message": "Invalid speed value"})

        elif msg_type == "save_state":
            try:
                await sim.save_state()
                await ws.send_json({"type": "ack", "action": "save_state"})
            except Exception as exc:
                await ws.send_json({"type": "error", "message": f"Save failed: {exc}"})

        else:
            await ws.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    # ------------------------------------------------------------------
    # Tick broadcast
    # ------------------------------------------------------------------

    async def broadcast_tick(self):
        """Build the current state snapshot and push it to every WebSocket client."""
        if not self.ws_clients:
            return

        payload = self._build_tick_state()
        raw = json.dumps(payload)

        stale: list[web.WebSocketResponse] = []
        for ws in set(self.ws_clients):
            try:
                await ws.send_str(raw)
            except Exception:
                stale.append(ws)

        for ws in stale:
            self.ws_clients.discard(ws)

    def _build_tick_state(self) -> dict:
        """Construct the full tick payload from simulation attributes."""
        sim = self.simulation

        # -- agents --
        agents_dict = {}
        for name, agent in sim.agents.items():
            activity = "idle"
            if agent.current_plan_item:
                activity = agent.current_plan_item.description
            agents_dict[name] = {
                "location": agent.current_location or "Unknown",
                "mood_valence": round(agent.mood_valence, 2),
                "mood_emoji": agent.mood_emoji,
                "sub_area": agent.current_sub_area or "",
                "activity": activity,
                "llm_active": (name == sim.display.current_llm_agent),
            }

        # -- locations --
        locations_dict = {}
        for loc_name, loc in sim.environment.locations.items():
            if loc.current_agents:
                locations_dict[loc_name] = sorted(loc.current_agents)

        # -- active conversations --
        active_convos = []
        for conv in sim.conversation_manager.active_conversations.values():
            active_convos.append({
                "agent1": conv.agent1,
                "agent2": conv.agent2,
                "location": conv.location,
                "turns": len(conv.turns),
            })

        # -- day calculation --
        day = 1
        if hasattr(sim.display, "sim_day"):
            day = sim.display.sim_day

        return {
            "type": "tick",
            "time": sim.current_time.isoformat(),
            "tick_count": sim.tick_count,
            "speed": sim.simulation_speed,
            "running": sim.running and not sim._paused,
            "day": day,
            "agents": agents_dict,
            "locations": locations_dict,
            "active_conversations": active_convos,
            "recent_events": list(sim.display.recent_events[-30:]),
            "stats": {
                "total_memories": sim.stats.get("total_memories", 0),
                "total_conversations": sim.stats.get("total_conversations", 0),
                "total_reflections": sim.stats.get("total_reflections", 0),
                "total_observations": sim.stats.get("total_observations", 0),
                "total_plans": sim.stats.get("total_plans", 0),
            },
            "llm": {
                "agent": sim.display.current_llm_agent or "",
                "task": sim.display.current_llm_task or "",
                "model": sim.display.current_llm_model or "",
                "call_count": sim.display.llm_call_count,
            },
        }

    # ------------------------------------------------------------------
    # REST: agent detail
    # ------------------------------------------------------------------

    async def handle_agent(self, request: web.Request) -> web.Response:
        """GET /api/agent/{name} -- full persona + current state."""
        name = request.match_info["name"]
        agent = self.simulation.agents.get(name)
        if agent is None:
            raise web.HTTPNotFound(text=json.dumps({"error": f"Agent '{name}' not found"}),
                                   content_type="application/json")

        activity = "idle"
        if agent.current_plan_item:
            activity = agent.current_plan_item.description

        daily_plan = []
        for item in agent.daily_plan:
            daily_plan.append({
                "description": item.description,
                "location": item.location,
                "start_time": item.start_time.isoformat(),
                "duration_minutes": item.duration_minutes,
                "completed": item.completed,
            })

        body = {
            "name": agent.name,
            "persona": agent.persona,
            "current_location": agent.current_location,
            "current_sub_area": agent.current_sub_area,
            "current_activity": activity,
            "daily_plan": daily_plan,
        }
        return web.json_response(body)

    # ------------------------------------------------------------------
    # REST: agent memories
    # ------------------------------------------------------------------

    async def handle_agent_memories(self, request: web.Request) -> web.Response:
        """GET /api/agent/{name}/memories?type=observation&limit=20"""
        name = request.match_info["name"]
        agent = self.simulation.agents.get(name)
        if agent is None:
            raise web.HTTPNotFound(text=json.dumps({"error": f"Agent '{name}' not found"}),
                                   content_type="application/json")

        memory_type = request.query.get("type", None)
        if memory_type in (None, "", "all"):
            memory_type = None
        try:
            limit = int(request.query.get("limit", 20))
        except ValueError:
            limit = 20
        try:
            offset = int(request.query.get("offset", 0))
        except ValueError:
            offset = 0

        # Fetch limit+offset and slice (MemoryStream.get_memories doesn't support offset)
        all_memories = agent.memory_stream.get_memories(limit=limit + offset, memory_type=memory_type)
        memories = all_memories[offset:]

        result = []
        for mem in memories:
            result.append({
                "id": mem.id,
                "description": mem.description,
                "memory_type": mem.memory_type,
                "importance_score": mem.importance_score,
                "creation_timestamp": mem.creation_timestamp.isoformat(),
                "last_access_timestamp": mem.last_access_timestamp.isoformat(),
                "location": mem.location,
                "source_memory_ids": mem.source_memory_ids or [],
            })

        return web.json_response({"agent": name, "count": len(result), "memories": result})

    # ------------------------------------------------------------------
    # REST: agent conversations
    # ------------------------------------------------------------------

    async def handle_agent_conversations(self, request: web.Request) -> web.Response:
        """GET /api/agent/{name}/conversations -- all conversations this agent participated in."""
        name = request.match_info["name"]
        if name not in self.simulation.agents:
            raise web.HTTPNotFound(text=json.dumps({"error": f"Agent '{name}' not found"}),
                                   content_type="application/json")

        cm = self.simulation.conversation_manager
        conversations = []

        # Active conversations involving this agent
        for conv in cm.active_conversations.values():
            if name in (conv.agent1, conv.agent2):
                conversations.append(self._serialize_conversation(conv, active=True))

        # Historical conversations involving this agent
        for conv in cm.conversation_history:
            if name in (conv.agent1, conv.agent2):
                conversations.append(self._serialize_conversation(conv, active=False))

        return web.json_response({"agent": name, "count": len(conversations), "conversations": conversations})

    # ------------------------------------------------------------------
    # REST: all conversations
    # ------------------------------------------------------------------

    async def handle_conversations(self, request: web.Request) -> web.Response:
        """GET /api/conversations -- all conversations with full transcripts."""
        cm = self.simulation.conversation_manager
        conversations = []

        for conv in cm.active_conversations.values():
            conversations.append(self._serialize_conversation(conv, active=True))

        for conv in cm.conversation_history:
            conversations.append(self._serialize_conversation(conv, active=False))

        return web.json_response({"count": len(conversations), "conversations": conversations})

    # ------------------------------------------------------------------
    # REST: locations
    # ------------------------------------------------------------------

    async def handle_locations(self, request: web.Request) -> web.Response:
        """GET /api/locations -- all locations with descriptions and occupants."""
        env = self.simulation.environment
        locations = []

        for loc_name, loc in env.locations.items():
            locations.append({
                "name": loc.name,
                "description": loc.description,
                "sub_areas": loc.sub_areas,
                "capacity": loc.capacity,
                "objects": loc.objects or [],
                "current_agents": sorted(loc.current_agents),
                "agent_count": len(loc.current_agents),
            })

        return web.json_response({"count": len(locations), "locations": locations})

    async def handle_pause(self, request: web.Request) -> web.Response:
        """POST /api/pause?yield_vram=true — pause sim, optionally unload GPU."""
        yield_vram = request.query.get("yield_vram", "false").lower() == "true"
        self.simulation.pause(yield_vram=yield_vram)
        return web.json_response({"status": "paused", "yield_vram": yield_vram})

    async def handle_resume(self, request: web.Request) -> web.Response:
        """POST /api/resume — resume sim, reload GPU if needed."""
        await self.simulation.resume()
        return web.json_response({"status": "resumed"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_conversation(conv, *, active: bool) -> dict:
        """Convert a Conversation object to a JSON-safe dict."""
        turns = []
        for turn in conv.turns:
            turns.append({
                "speaker": turn.speaker,
                "message": turn.message,
                "timestamp": turn.timestamp.isoformat(),
            })
        return {
            "agent1": conv.agent1,
            "agent2": conv.agent2,
            "location": conv.location,
            "start_time": conv.start_time.isoformat(),
            "active": active,
            "turn_count": len(conv.turns),
            "turns": turns,
        }
