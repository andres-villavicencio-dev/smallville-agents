"""Rich terminal display for the generative agents simulation."""
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text


class SimulationDisplay:
    """Rich terminal display for the simulation."""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.live = None
        
        # Simulation state
        self.simulation_time = datetime.now()
        self.simulation_speed = 10
        self.tick_count = 0
        self.sim_day = 1
        self.start_time = datetime.now()
        self.real_start_time = datetime.now()
        
        # Agent data
        self.agent_activities = {}
        self.agent_count = 0
        
        # Events & conversations
        self.recent_events = []
        self.location_populations = {}
        self.conversation_summaries = []
        self.conversation_log = []  # Full conversation history
        
        # LLM tracking
        self.current_llm_agent = ""
        self._last_llm_update = None
        self.current_llm_task = ""
        self.current_llm_model = ""
        self.llm_call_count = 0
        self.llm_calls_per_model: dict[str, int] = {}
        
        # Stats
        self.total_memories = 0
        self.total_conversations = 0
        self.total_reflections = 0
        self.total_observations = 0
        self.committee_mode = False
        
        self._setup_layout()
        # Pre-populate with initial content so Rich doesn't show raw Layout placeholders
        self._populate_initial_content()
    
    def _setup_layout(self):
        """Set up the terminal layout."""
        self.layout.split_column(
            Layout(name="header", size=5),
            Layout(name="main", ratio=1),
            Layout(name="status_bar", size=4),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2)
        )
        
        self.layout["left"].split_column(
            Layout(name="agents", ratio=2),
            Layout(name="events", ratio=1)
        )
        
        self.layout["right"].split_column(
            Layout(name="locations", ratio=1),
            Layout(name="conversations", ratio=1)
        )
    
    def _populate_initial_content(self):
        """Fill all layout sections with initial panels so Rich doesn't show placeholders."""
        self.layout["header"].update(self._create_header())
        self.layout["agents"].update(Panel("Initializing agents...", title="🤖 Agent Activities", border_style="cyan"))
        self.layout["events"].update(Panel("Waiting for events...", title="📝 Recent Events", border_style="green"))
        self.layout["locations"].update(Panel("Loading locations...", title="🏢 Locations", border_style="yellow"))
        self.layout["conversations"].update(Panel("No conversations yet...", title="💬 Conversations", border_style="magenta"))
        self.layout["status_bar"].update(Panel("🧠 LLM: starting up...", title="⚙️ System Status", border_style="bright_black"))
        self.layout["footer"].update(self._create_footer())
    
    def start_display(self):
        """Start the live display (no-op if TUI disabled)."""
        self.real_start_time = datetime.now()
        if getattr(self, '_tui_disabled', False):
            return
        self.live = Live(self.layout, refresh_per_second=2, console=self.console)
        self.live.start()

    def disable_tui(self):
        """Disable the Rich TUI entirely (WebUI-only mode)."""
        self._tui_disabled = True

    def stop_display(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()
    
    # ── Update Methods ──────────────────────────────────────────────────
    
    def update_simulation_time(self, sim_time: datetime, speed: int, tick: int):
        """Update simulation time information."""
        self.simulation_time = sim_time
        self.simulation_speed = speed
        self.tick_count = tick
        # Calculate sim day
        if hasattr(self, 'start_time') and self.start_time:
            delta = sim_time - self.start_time
            self.sim_day = delta.days + 1
    
    def set_start_time(self, start_time: datetime):
        """Set the simulation start time for day calculation."""
        self.start_time = start_time
    
    def set_committee_mode(self, enabled: bool):
        """Set whether committee mode is active."""
        self.committee_mode = enabled
    
    def update_agent_activities(self, activities: dict[str, str]):
        """Update agent activities."""
        self.agent_activities = activities.copy()
        self.agent_count = len(activities)
    
    def add_event(self, event: str, timestamp: datetime | None = None):
        """Add an event to the recent events list."""
        if timestamp is None:
            timestamp = self.simulation_time
        time_str = timestamp.strftime("%H:%M")
        self.recent_events.append(f"[{time_str}] {event}")
        if len(self.recent_events) > 30:
            self.recent_events = self.recent_events[-30:]
    
    def update_location_populations(self, populations: dict[str, list[str]]):
        """Update location population data."""
        self.location_populations = populations.copy()
    
    def update_conversations(self, conversation_summaries: list[str]):
        """Update active conversations."""
        self.conversation_summaries = conversation_summaries.copy()
    
    def add_conversation_line(self, speaker: str, target: str, message: str):
        """Add a conversation line to the log."""
        time_str = self.simulation_time.strftime("%H:%M")
        self.conversation_log.append(f"[{time_str}] {speaker} → {target}: {message}")
        if len(self.conversation_log) > 50:
            self.conversation_log = self.conversation_log[-50:]
    
    def update_llm_status(self, agent: str = "", task: str = "", model: str = ""):
        """Update current LLM call status."""
        if agent:
            self.current_llm_agent = agent
            self.current_llm_task = task
            self.current_llm_model = model
            self._last_llm_update = datetime.now()
            self.llm_call_count += 1
            if model:
                self.llm_calls_per_model[model] = self.llm_calls_per_model.get(model, 0) + 1
            logger.debug(f"LLM status: {agent} → {task} [{model}] (call #{self.llm_call_count})")
    
    def update_stats(self, stats: dict[str, Any]):
        """Update simulation statistics."""
        self.total_memories = stats.get("total_memories", self.total_memories)
        self.total_conversations = stats.get("total_conversations", self.total_conversations)
        self.total_reflections = stats.get("total_reflections", self.total_reflections)
        self.total_observations = stats.get("total_observations", self.total_observations)
    
    # ── Panel Builders ──────────────────────────────────────────────────
    
    def _create_header(self) -> Panel:
        """Create the header panel with simulation info."""
        sim_time_str = self.simulation_time.strftime("%A, %B %d, %Y — %H:%M:%S")
        real_elapsed = datetime.now() - self.real_start_time
        elapsed_str = str(real_elapsed).split('.')[0]  # Remove microseconds
        
        # Time of day indicator
        hour = self.simulation_time.hour
        if 6 <= hour < 12:
            tod = "🌅 Morning"
        elif 12 <= hour < 17:
            tod = "☀️ Afternoon"
        elif 17 <= hour < 21:
            tod = "🌆 Evening"
        else:
            tod = "🌙 Night"
        
        header = Text()
        header.append("🏘️  SMALLVILLE", style="bold cyan")
        mode_str = " [Committee Mode]" if self.committee_mode else " [Single Model]"
        header.append(mode_str, style="bold magenta" if self.committee_mode else "dim")
        header.append(f"\n📅 {sim_time_str}  {tod}", style="white")
        header.append(f"\n📊 Day {self.sim_day}", style="yellow")
        header.append(f"  │  Tick {self.tick_count}", style="green")
        header.append(f"  │  Speed {self.simulation_speed}x", style="cyan")
        header.append(f"  │  Agents {self.agent_count}", style="blue")
        header.append(f"  │  ⏱ Real: {elapsed_str}", style="dim white")
        
        return Panel(header, border_style="bright_blue")
    
    def _create_agents_panel(self) -> Panel:
        """Create the agents activity panel."""
        if not self.agent_activities:
            return Panel("Waiting for agents...", title="🤖 Agent Activities")
        
        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
        table.add_column("Agent", style="cyan", width=22)
        table.add_column("📍", style="yellow", width=22)
        table.add_column("Activity", style="white")
        
        for agent_name in sorted(self.agent_activities.keys()):
            activity_info = self.agent_activities[agent_name]
            
            if " | " in activity_info:
                location, activity = activity_info.split(" | ", 1)
            else:
                location = "?"
                activity = activity_info
            
            # Truncate
            if len(activity) > 45:
                activity = activity[:42] + "..."
            if len(location) > 20:
                location = location[:17] + "..."
            
            # Highlight if this agent is currently using the LLM
            name_style = "bold bright_green" if agent_name == self.current_llm_agent else "cyan"
            indicator = "⚡" if agent_name == self.current_llm_agent else "  "
            
            table.add_row(
                Text(f"{indicator}{agent_name}", style=name_style),
                location,
                activity
            )
        
        return Panel(table, title=f"🤖 Agent Activities ({self.agent_count})", border_style="cyan")
    
    def _create_events_panel(self) -> Panel:
        """Create the recent events panel."""
        if not self.recent_events:
            return Panel("Waiting for events...", title="📝 Recent Events")
        
        events_text = Text()
        for event in self.recent_events[-12:]:
            events_text.append(event + "\n", style="white")
        
        return Panel(events_text, title=f"📝 Recent Events ({len(self.recent_events)})", border_style="green")
    
    def _create_locations_panel(self) -> Panel:
        """Create the locations panel."""
        if not self.location_populations:
            return Panel("No location data", title="🏢 Locations")
        
        table = Table(show_header=True, header_style="bold yellow", box=None, padding=(0, 1))
        table.add_column("Location", style="yellow", width=20)
        table.add_column("#", style="bold white", width=3, justify="center")
        table.add_column("Agents", style="cyan")
        
        # Sort by population (busiest first)
        sorted_locs = sorted(
            self.location_populations.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for location, agents in sorted_locs:
            count = len(agents)
            agent_list = ", ".join(a.split()[0] for a in agents[:4])  # First names only
            if len(agents) > 4:
                agent_list += f" +{len(agents)-4}"
            
            # Color code by density
            if count >= 5:
                count_style = "bold red"
            elif count >= 3:
                count_style = "bold yellow"
            else:
                count_style = "white"
            
            loc_name = location[:18] if len(location) > 18 else location
            table.add_row(loc_name, Text(str(count), style=count_style), agent_list)
        
        total_placed = sum(len(a) for a in self.location_populations.values())
        return Panel(table, title=f"🏢 Locations ({total_placed} agents placed)", border_style="yellow")
    
    def _create_conversations_panel(self) -> Panel:
        """Create the conversations panel with both active and recent log."""
        content = Text()
        
        # Active conversations
        if self.conversation_summaries:
            content.append("🔴 ACTIVE:\n", style="bold red")
            for summary in self.conversation_summaries:
                content.append(f"  • {summary}\n", style="bright_white")
            content.append("\n")
        
        # Recent conversation log
        if self.conversation_log:
            content.append("📜 Recent:\n", style="bold dim")
            for line in self.conversation_log[-8:]:
                content.append(f"  {line}\n", style="white")
        
        if not self.conversation_summaries and not self.conversation_log:
            content.append("No conversations yet...\n", style="dim")
            content.append("Agents need to be at the same\nlocation to start talking.", style="dim italic")
        
        return Panel(
            content,
            title=f"💬 Conversations ({self.total_conversations} total)",
            border_style="magenta"
        )
    
    def _create_status_bar(self) -> Panel:
        """Create the status bar with LLM info and stats."""
        status = Text()
        
        # LLM status line
        # Auto-clear LLM status after 30s of no updates
        if self._last_llm_update and (datetime.now() - self._last_llm_update).seconds > 30:
            is_active = False
        else:
            is_active = bool(self.current_llm_agent)
        
        if is_active:
            status.append("🧠 LLM: ", style="bold")
            status.append(self.current_llm_agent, style="bright_green")
            if self.current_llm_task:
                status.append(f" → {self.current_llm_task}", style="yellow")
            if self.current_llm_model:
                status.append(f" [{self.current_llm_model}]", style="dim cyan")
        elif self.llm_call_count > 0:
            status.append("🧠 LLM: ", style="bold")
            status.append("idle", style="dim")
            status.append(f" (last: {self.current_llm_agent})", style="dim")
        else:
            status.append("🧠 LLM: ", style="bold")
            status.append("waiting for first call...", style="dim yellow")
        
        status.append(f"  │  Calls: {self.llm_call_count}", style="dim white")
        
        # Model usage breakdown
        if self.llm_calls_per_model:
            status.append("\n📊 Models: ", style="bold")
            for i, (model, count) in enumerate(sorted(self.llm_calls_per_model.items(), key=lambda x: -x[1])):
                if i > 0:
                    status.append("  ", style="dim")
                short_name = model.split(":")[-1] if ":" in model else model
                short_name = model.split("/")[-1] if "/" in short_name else short_name
                status.append(f"{model}={count}", style="dim cyan")
        
        # Stats line
        status.append(f"\n💾 Memories: {self.total_memories}", style="dim white")
        status.append(f"  │  Observations: {self.total_observations}", style="dim white")
        status.append(f"  │  Reflections: {self.total_reflections}", style="dim white")
        status.append(f"  │  Conversations: {self.total_conversations}", style="dim white")
        
        return Panel(status, title="⚙️ System Status", border_style="bright_black")
    
    def _create_footer(self) -> Panel:
        """Create the footer panel."""
        footer_text = Text()
        footer_text.append("Controls: ", style="bold white")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" stop  │  ", style="dim")
        footer_text.append("Logs: simulation.log", style="green")
        footer_text.append("  │  ", style="dim")
        footer_text.append("⚡= currently using LLM", style="bright_green")
        
        return Panel(Align.center(footer_text), style="bright_black", border_style="dim")
    
    def refresh_display(self):
        """Refresh the entire display."""
        if not self.live:
            return
        try:
            self.layout["header"].update(self._create_header())
            self.layout["agents"].update(self._create_agents_panel())
            self.layout["events"].update(self._create_events_panel())
            self.layout["locations"].update(self._create_locations_panel())
            self.layout["conversations"].update(self._create_conversations_panel())
            self.layout["status_bar"].update(self._create_status_bar())
            self.layout["footer"].update(self._create_footer())
        except Exception:
            pass  # Don't crash simulation on display errors
    
    # ── Print Methods ───────────────────────────────────────────────────
    
    def print_startup_message(self):
        """Print a startup message."""
        self.console.print()
        self.console.print("🏘️  [bold cyan]Smallville — Generative Agents Simulation[/bold cyan]")
        self.console.print("[yellow]Based on 'Generative Agents: Interactive Simulacra of Human Behavior'[/yellow]")
        self.console.print("[dim]Park et al., 2023 (arXiv:2304.03442)[/dim]")
        if self.committee_mode:
            self.console.print("[bold magenta]🧠 Committee of Experts mode enabled[/bold magenta]")
        self.console.print("[green]Starting simulation...[/green]")
        self.console.print()
    
    def print_shutdown_message(self):
        """Print a shutdown message."""
        self.console.print()
        self.console.print("[yellow]Simulation stopped[/yellow]")
        
        # Print final stats
        elapsed = datetime.now() - self.real_start_time
        self.console.print(f"[cyan]Duration: {str(elapsed).split('.')[0]} real time[/cyan]")
        self.console.print(f"[cyan]Ticks: {self.tick_count} | Day: {self.sim_day}[/cyan]")
        self.console.print(f"[cyan]LLM calls: {self.llm_call_count}[/cyan]")
        if self.llm_calls_per_model:
            for model, count in sorted(self.llm_calls_per_model.items(), key=lambda x: -x[1]):
                self.console.print(f"  [dim]{model}: {count} calls[/dim]")
        self.console.print(f"[cyan]Memories: {self.total_memories} | Conversations: {self.total_conversations} | Reflections: {self.total_reflections}[/cyan]")
        self.console.print("[green]State saved. Thank you for using Smallville![/green]")
    
    def print_error(self, error_msg: str):
        self.console.print(f"[bold red]Error: {error_msg}[/bold red]")
    
    def print_warning(self, warning_msg: str):
        self.console.print(f"[bold yellow]Warning: {warning_msg}[/bold yellow]")
    
    def print_info(self, info_msg: str):
        self.console.print(f"[cyan]Info: {info_msg}[/cyan]")
    
    def show_agent_details(self, agent_name: str, memories: list[str], 
                          current_plan: str, reflections: list[str]):
        """Show detailed information about an agent."""
        self.console.print(f"\n[bold cyan]🤖 {agent_name} Details[/bold cyan]")
        self.console.print(f"[yellow]Current Plan:[/yellow] {current_plan}")
        if memories:
            self.console.print("\n[green]Recent Memories:[/green]")
            for i, memory in enumerate(memories[-5:], 1):
                self.console.print(f"  {i}. {memory}")
        if reflections:
            self.console.print("\n[magenta]Recent Reflections:[/magenta]")
            for i, reflection in enumerate(reflections[-3:], 1):
                self.console.print(f"  {i}. {reflection}")
        self.console.print()
    
    def show_simulation_stats(self, stats: dict[str, Any]):
        """Show simulation statistics."""
        self.console.print("\n[bold cyan]📊 Simulation Statistics[/bold cyan]")
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="yellow")
        table.add_column("Value", style="white")
        for key, value in stats.items():
            table.add_row(key, str(value))
        self.console.print(table)
        self.console.print()


class ProgressDisplay:
    """Simple progress display for initialization."""
    
    def __init__(self):
        self.console = Console()
        self.progress = None
    
    def start_progress(self, description: str):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )
        self.progress.start()
        return self.progress.add_task(description)
    
    def update_progress(self, task_id, description: str):
        if self.progress:
            self.progress.update(task_id, description=description)
    
    def stop_progress(self):
        if self.progress:
            self.progress.stop()
            self.progress = None
