"""Main simulation loop for Generative Agents."""
import asyncio
import argparse
import signal
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

from agent import GenerativeAgent
from environment import SmallvilleEnvironment
from conversation import ConversationManager
from display import SimulationDisplay, ProgressDisplay
from personas import get_all_agent_names, get_agents_by_location, select_agent_subset
import config as cfg
from config import (
    DEFAULT_SIMULATION_SPEED, TICK_DURATION_SECONDS, DEFAULT_SIM_DAYS, DEFAULT_NUM_AGENTS,
    START_DATE, START_TIME, get_config
)
from llm import set_llm_status_callback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SmallvilleSimulation:
    """Main simulation controller."""
    
    def __init__(self, use_gpu_queue: bool = True, speed: int = DEFAULT_SIMULATION_SPEED, num_agents: int = DEFAULT_NUM_AGENTS):
        self.agents: Dict[str, GenerativeAgent] = {}
        self.environment = SmallvilleEnvironment()
        self.conversation_manager = ConversationManager()
        self.display = SimulationDisplay()
        
        # Wire LLM status callback to display
        set_llm_status_callback(lambda agent, task, model: self.display.update_llm_status(agent, task, model))
        
        self.num_agents = max(5, min(num_agents, 25))
        self.use_gpu_queue = use_gpu_queue
        self.simulation_speed = speed
        self.tick_duration = TICK_DURATION_SECONDS
        self.running = False
        self.current_time = self._parse_start_time()
        self.tick_count = 0
        self.save_state_interval = 100  # Save state every 100 ticks
        
        # Statistics
        self.stats = {
            "total_memories": 0,
            "total_conversations": 0,
            "total_reflections": 0,
            "total_observations": 0,
            "total_plans": 0
        }
        self._last_reflection_count = 0

        # Web UI
        self.webui = None
        self._paused = False

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _parse_start_time(self) -> datetime:
        """Parse start date and time from config."""
        try:
            date_str = f"{START_DATE} {START_TIME}"
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        except ValueError:
            logger.warning(f"Invalid start time format, using current time")
            return datetime.now()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        if not self.running:
            # Second Ctrl+C — force exit immediately
            logger.info("Forced shutdown")
            import os
            os._exit(1)
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        # Tell committee to abort mid-pipeline
        if cfg.USE_COMMITTEE:
            try:
                from committee import get_committee
                get_committee().shutdown()
            except Exception:
                pass

    def pause(self):
        """Pause the simulation."""
        self._paused = True
        logger.info("Simulation paused")

    def resume(self):
        """Resume the simulation."""
        self._paused = False
        logger.info("Simulation resumed")

    def set_speed(self, speed: int):
        """Set simulation speed."""
        self.simulation_speed = max(1, min(100, speed))
        logger.info(f"Simulation speed set to {self.simulation_speed}x")

    async def save_state(self):
        """Public save state method for web UI."""
        await self._save_state()
    
    async def initialize(self):
        """Initialize the simulation."""
        progress = ProgressDisplay()
        
        try:
            task = progress.start_progress("Initializing simulation...")
            
            # Create all agents
            progress.update_progress(task, "Creating agents...")
            if self.num_agents < 25:
                agent_names = select_agent_subset(self.num_agents)
                logger.info(f"Running with {len(agent_names)} agents: {', '.join(agent_names)}")
            else:
                agent_names = get_all_agent_names()
            
            for name in agent_names:
                agent = GenerativeAgent(name)
                self.agents[name] = agent
                
                # Place agents at their home locations initially
                from personas import get_agent_persona
                persona = get_agent_persona(name)
                home = persona.get("home_location", "Lin Family Home")
                self.environment.move_agent(name, home)
                agent.move_to_location(home)
            
            # Try to load saved state (auto-resume)
            state_file = os.path.join("saves", "latest_state.json")
            loaded_state = False
            if os.path.exists(state_file):
                progress.update_progress(task, "Resuming from saved state...")
                try:
                    await self.load_state(state_file)
                    loaded_state = True
                    # Prune phantom agents from a larger save
                    active_names = set(self.agents.keys())
                    for loc in self.environment.locations.values():
                        loc.current_agents &= active_names
                    self.environment.agent_locations = {
                        k: v for k, v in self.environment.agent_locations.items() if k in active_names
                    }
                    self.environment.agent_sub_areas = {
                        k: v for k, v in self.environment.agent_sub_areas.items() if k in active_names
                    }
                    logger.info(f"Resumed from saved state: tick {self.tick_count}, time {self.current_time}")
                except Exception as e:
                    logger.warning(f"Failed to load state, regenerating plans: {e}")
            
            # Generate daily plans only for agents that don't have one
            agents_needing_plans = [a for a in self.agents.values() if not a.daily_plan]
            
            if agents_needing_plans:
                progress.update_progress(task, f"Generating daily plans ({len(agents_needing_plans)} agents)...")
                plan_tasks = [a.plan_daily_schedule(self.current_time) for a in agents_needing_plans]
                await asyncio.gather(*plan_tasks)
            else:
                progress.update_progress(task, f"Resumed {len(self.agents)} agents from saved state")
            
            progress.update_progress(task, "Setting up display...")
            
            # Initialize display
            self.display.set_start_time(self.current_time)
            self.display.set_committee_mode(cfg.USE_COMMITTEE)
            self.display.update_simulation_time(self.current_time, self.simulation_speed, 0)
            
            progress.stop_progress()
            
            logger.info(f"Simulation initialized with {len(self.agents)} agents")
            self._update_display_data()
            
        except Exception as e:
            progress.stop_progress()
            raise e
    
    async def run(self, duration_days: int = DEFAULT_SIM_DAYS):
        """Run the simulation for the specified duration."""
        self.running = True
        end_time = self.current_time + timedelta(days=duration_days)
        
        self.display.print_startup_message()
        
        # Seed initial events
        self.display.add_event(f"🏘️ Simulation started — {len(self.agents)} agents in Smallville")
        self.display.add_event(f"📅 Sim date: {self.current_time.strftime('%A, %B %d, %Y %H:%M')}")
        mode = "Committee of Experts" if cfg.USE_COMMITTEE else "Single Model"
        self.display.add_event(f"🧠 Mode: {mode}")
        
        # Show initial agent placement
        for loc_name, loc in self.environment.locations.items():
            if loc.current_agents:
                names = ", ".join(list(loc.current_agents)[:3])
                extra = f" +{len(loc.current_agents)-3}" if len(loc.current_agents) > 3 else ""
                self.display.add_event(f"📍 {loc_name}: {names}{extra}")
        
        # Populate display with initial data BEFORE starting Live
        self._update_display_data()
        self.display.refresh_display()
        self.display.start_display()
        
        try:
            while self.running and self.current_time < end_time:
                # Handle pause
                while self._paused and self.running:
                    await asyncio.sleep(0.1)
                    if self.webui:
                        await self.webui.broadcast_tick()

                # Update display before the step (so it shows current state while LLM runs)
                self._update_display_data()
                self.display.refresh_display()
                
                # Periodic status event every 20 ticks
                if self.tick_count % 20 == 0 and self.tick_count > 0:
                    active_convos = len(self.conversation_manager.active_conversations)
                    busiest = ""
                    if self.display.location_populations:
                        busiest_loc = max(self.display.location_populations.items(), key=lambda x: len(x[1]), default=("", []))
                        if busiest_loc[1]:
                            busiest = f" | Busiest: {busiest_loc[0]} ({len(busiest_loc[1])})"
                    total_skills = sum(a.skill_bank.get_stats()["total"] for a in self.agents.values())
                    skills_str = f" | Skills: {total_skills}" if total_skills else ""
                    self.display.add_event(
                        f"⏱ Tick {self.tick_count} | Memories: {self.stats.get('total_memories', 0)} | Convos: {active_convos} active{busiest}{skills_str}"
                    )
                
                # Run one simulation step
                await self._simulation_step()
                
                # Update display after the step
                self._update_display_data()
                self.display.refresh_display()

                # Broadcast to web UI clients
                if self.webui:
                    await self.webui.broadcast_tick()

                # Save state periodically
                if self.tick_count % self.save_state_interval == 0:
                    await self._save_state()
                
                # Sleep for real-time simulation speed
                await asyncio.sleep(1.0 / self.simulation_speed)
                
                self.tick_count += 1
                self.current_time += timedelta(seconds=self.tick_duration)
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            self.display.print_error(str(e))
        finally:
            self.running = False
            await self._shutdown()
    
    async def _simulation_step(self):
        """Execute one simulation step."""
        try:
            # 1. Update agent activities based on plans
            for agent in self.agents.values():
                new_activity = agent.update_current_activity(self.current_time)
                
                # Move agent to planned location if needed
                if agent.current_plan_item:
                    planned_location = agent.current_plan_item.location
                    current_location = agent.current_location
                    
                    if planned_location != current_location:
                        success = self.environment.move_agent(agent.name, planned_location)
                        if success:
                            agent.move_to_location(planned_location)
                            self.display.add_event(
                                f"🚶 {agent.name} → {planned_location}"
                            )
                
                # Log new activities as events
                if new_activity:
                    self.display.add_event(f"📋 {agent.name}: {new_activity[:60]}")
            
            if not self.running:
                return

            # 2. Generate environmental observations (sample a few agents per tick, not all)
            import random
            agents_to_observe = random.sample(
                list(self.agents.values()),
                min(5, len(self.agents))
            )
            observation_tasks = []
            for agent in agents_to_observe:
                observations = self.environment.observe_environment(agent.name)
                for obs in observations:
                    observation_tasks.append(agent.observe(obs, agent.current_location))

            if observation_tasks:
                await asyncio.gather(*observation_tasks, return_exceptions=True)

            if not self.running:
                return

            # 3. Check for conversation opportunities
            await self._update_conversations()
            
            # 4. Process any triggered reflections (handled automatically in agent.observe())
        
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
    
    async def _update_conversations(self):
        """Update conversations between agents."""
        try:
            # Check for new conversation opportunities
            agent_pairs_checked = set()
            
            for location_name, location in self.environment.locations.items():
                agents_here = list(location.current_agents)
                
                if len(agents_here) >= 2:
                    # Check all pairs of agents at this location
                    for i, agent1 in enumerate(agents_here):
                        for agent2 in agents_here[i+1:]:
                            pair_key = tuple(sorted([agent1, agent2]))
                            
                            if pair_key in agent_pairs_checked:
                                continue
                            agent_pairs_checked.add(pair_key)
                            
                            # Skip if already in conversation or on cooldown
                            if self.conversation_manager.has_active_conversation(agent1, agent2):
                                continue
                            if self.conversation_manager.is_on_cooldown(agent1, agent2, self.tick_count):
                                continue
                            
                            # Check if conversation should start
                            context = f"Both at {location_name}"
                            agent1_memory = self.agents[agent1].memory_stream
                            
                            logger.info(f"Checking conversation: {agent1} <-> {agent2} at {location_name}")
                            should_talk = await self.conversation_manager.should_initiate_conversation(
                                agent1, agent2, context, agent1_memory
                            )
                            
                            if should_talk:
                                # Start conversation
                                memory_streams = {
                                    name: agent.memory_stream 
                                    for name, agent in self.agents.items()
                                }
                                
                                conversation = await self.conversation_manager.start_conversation(
                                    agent1, agent2, location_name, memory_streams
                                )
                                
                                if conversation:
                                    self.display.add_event(
                                        f"💬 {agent1} started talking to {agent2} at {location_name}"
                                    )
                                    # Log the opening line
                                    if conversation.turns:
                                        t = conversation.turns[0]
                                        self.display.add_conversation_line(t.speaker, agent2, t.message[:80])
                                    self.stats["total_conversations"] += 1
            
            # Update existing conversations
            memory_streams = {
                name: agent.memory_stream 
                for name, agent in self.agents.items()
            }
            
            # Track turn count before update to detect new lines
            turn_counts = {}
            for conv in self.conversation_manager.active_conversations.values():
                key = (conv.agent1, conv.agent2)
                turn_counts[key] = len(conv.turns)
            
            skill_banks = {
                name: agent.skill_bank
                for name, agent in self.agents.items()
            }
            await self.conversation_manager.update_conversations(
                memory_streams, skill_banks, 
                agents=self.agents, current_time=self.current_time,
                current_tick=self.tick_count
            )
            
            # Log any new conversation lines to display
            for conv in self.conversation_manager.active_conversations.values():
                key = (conv.agent1, conv.agent2)
                old_count = turn_counts.get(key, 0)
                for turn in conv.turns[old_count:]:
                    other = conv.agent2 if turn.speaker == conv.agent1 else conv.agent1
                    self.display.add_conversation_line(turn.speaker, other, turn.message[:80])
            
            # Also log from recently finished conversations
            for conv in self.conversation_manager.conversation_history[-3:]:
                if not conv.active and conv.turns:
                    last = conv.turns[-1]
                    # Only add if we haven't seen it (check by matching last turn)
                    pass  # conversation_history lines already logged during active phase
        
        except Exception as e:
            logger.error(f"Error updating conversations: {e}")
    
    def _update_display_data(self):
        """Update data for the display."""
        try:
            # Update simulation time
            self.display.update_simulation_time(
                self.current_time, self.simulation_speed, self.tick_count
            )
            
            # Update agent activities
            activities = {}
            for agent in self.agents.values():
                location = agent.current_location or "Unknown"
                activity = "idle"
                
                if agent.current_plan_item:
                    activity = agent.current_plan_item.description
                
                activities[agent.name] = f"{location} | {activity}"
            
            self.display.update_agent_activities(activities)
            
            # Update location populations
            populations = {}
            for location_name, location in self.environment.locations.items():
                if location.current_agents:
                    populations[location_name] = list(location.current_agents)
            
            self.display.update_location_populations(populations)
            
            # Update conversations
            conv_summaries = self.conversation_manager.get_active_conversations_summary()
            self.display.update_conversations(conv_summaries)
            
            # Update stats
            self._update_stats()
        
        except Exception as e:
            logger.error(f"Error updating display data: {e}")
    
    def _update_stats(self):
        """Update simulation statistics."""
        try:
            total_memories = 0
            total_reflections = 0
            
            # Only recount from DB every 10 ticks to avoid performance hit
            if self.tick_count % 10 == 0:
                for agent in self.agents.values():
                    memories = agent.memory_stream.get_memories()
                    total_memories += len(memories)
                    
                    reflections = agent.memory_stream.get_memories(memory_type="reflection")
                    total_reflections += len(reflections)
            else:
                total_memories = self.stats.get("total_memories", 0)
                total_reflections = self.stats.get("total_reflections", 0)
            
            total_observations = total_memories - total_reflections
            total_convos = len(self.conversation_manager.conversation_history)
            
            self.stats.update({
                "total_memories": total_memories,
                "total_reflections": total_reflections,
                "total_conversations": total_convos,
                "total_observations": total_observations,
                "total_plans": len([a for a in self.agents.values() if a.daily_plan])
            })
            
            # Detect new reflections and add as events
            if total_reflections > self._last_reflection_count:
                new_count = total_reflections - self._last_reflection_count
                # Find agents who just reflected
                for agent in self.agents.values():
                    refs = agent.memory_stream.get_memories(memory_type="reflection")
                    if refs:
                        latest = refs[-1]
                        desc = latest.description[:70]
                        self.display.add_event(f"🪞 {agent.name} reflected: {desc}...")
                self._last_reflection_count = total_reflections
            
            # Feed to display
            self.display.update_stats({
                "total_memories": total_memories,
                "total_observations": total_observations,
                "total_reflections": total_reflections,
                "total_conversations": total_convos,
            })
        
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    async def _save_state(self):
        """Save simulation state to file."""
        try:
            state = {
                "current_time": self.current_time.isoformat(),
                "tick_count": self.tick_count,
                "stats": self.stats,
                "environment": self.environment.get_environment_state(),
                "agents": {
                    name: agent.get_state()
                    for name, agent in self.agents.items()
                }
            }
            
            # Save to file
            os.makedirs("saves", exist_ok=True)
            save_file = f"saves/simulation_state_{self.current_time.strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(save_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Keep only latest save file
            latest_save = "saves/latest_state.json"
            with open(latest_save, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    async def load_state(self, state_file: str):
        """Load simulation state from file."""
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.current_time = datetime.fromisoformat(state["current_time"])
            self.tick_count = state["tick_count"]
            self.stats = state.get("stats", self.stats)
            
            # Load environment state
            self.environment.load_environment_state(state["environment"])
            
            # Load agent states
            for name, agent_state in state["agents"].items():
                if name in self.agents:
                    self.agents[name].load_state(agent_state)
            
            logger.info(f"Loaded state from {state_file}")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    async def _shutdown(self):
        """Clean shutdown of the simulation with timeout guard."""
        try:
            await asyncio.wait_for(self._shutdown_inner(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Shutdown timed out after 10s, forcing exit")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def _shutdown_inner(self):
        """Inner shutdown logic."""
        # Stop web UI
        if self.webui:
            await self.webui.stop()

        # Save final state
        await self._save_state()

        # Stop display
        self.display.stop_display()

        # Print final statistics
        self.display.show_simulation_stats(self.stats)
        self.display.print_shutdown_message()

        logger.info("Simulation shut down successfully")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Smallville Generative Agents Simulation")
    parser.add_argument("--speed", type=int, default=DEFAULT_SIMULATION_SPEED,
                       help="Simulation speed (game seconds per real second)")
    parser.add_argument("--days", type=int, default=DEFAULT_SIM_DAYS,
                       help="Number of days to simulate")
    parser.add_argument("--no-gpu-queue", action="store_true",
                       help="Don't use GPU queue for LLM calls")
    parser.add_argument("--load-state", type=str,
                       help="Load simulation state from file")
    parser.add_argument("--committee", action="store_true",
                       help="Use committee of experts (mixture of small models)")
    parser.add_argument("--num-agents", type=int, default=DEFAULT_NUM_AGENTS,
                       help="Number of agents to simulate (5-25, default: 25)")
    parser.add_argument("--config", action="store_true",
                       help="Show configuration and exit")
    parser.add_argument("--webui", action="store_true",
                       help="Enable web UI dashboard")
    parser.add_argument("--webui-port", type=int, default=8080,
                       help="Web UI port (default: 8080)")
    
    args = parser.parse_args()
    
    # Enable committee mode via flag
    if args.committee:
        import config as cfg
        cfg.USE_COMMITTEE = True
        # Re-import so modules pick up the change
        os.environ["USE_COMMITTEE"] = "1"
    
    if args.config:
        print("Current configuration:")
        config = get_config()
        for key, value in config.items():
            print(f"  {key}: {value}")
        return
    
    # Print mode
    if cfg.USE_COMMITTEE:
        from committee import print_committee_config
        print_committee_config()
    else:
        from llm import print_model_routing
        print_model_routing()
    
    # Create simulation
    simulation = SmallvilleSimulation(
        use_gpu_queue=not args.no_gpu_queue,
        speed=args.speed,
        num_agents=args.num_agents
    )
    
    try:
        # Initialize
        await simulation.initialize()

        # Start web UI if enabled
        if args.webui:
            from webui import SmallvilleWebUI
            simulation.webui = SmallvilleWebUI(simulation)
            await simulation.webui.start(port=args.webui_port)
            print(f"\n🌐 Web UI available at http://localhost:{args.webui_port}\n")

        # Load state if specified
        if args.load_state:
            await simulation.load_state(args.load_state)
        
        # Run simulation
        await simulation.run(duration_days=args.days)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\nSimulation interrupted")
        sys.exit(0)