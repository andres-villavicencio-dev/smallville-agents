"""Smallville environment implementation."""
import json
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from config import SMALLVILLE_LOCATIONS

logger = logging.getLogger(__name__)

@dataclass
class Location:
    """A location in Smallville."""
    name: str
    sub_areas: List[str]
    description: str = ""
    capacity: int = 10  # Maximum agents that can be here
    current_agents: Set[str] = None
    objects: List[str] = None
    
    def __post_init__(self):
        if self.current_agents is None:
            self.current_agents = set()
        if self.objects is None:
            self.objects = []

class SmallvilleEnvironment:
    """The Smallville environment where agents live and interact."""
    
    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.agent_locations: Dict[str, str] = {}  # agent_name -> location_name
        self.agent_sub_areas: Dict[str, str] = {}  # agent_name -> sub_area
        self.location_connections: Dict[str, List[str]] = {}
        self._initialize_locations()
        self._initialize_connections()
    
    def _initialize_locations(self):
        """Initialize all Smallville locations."""
        # Create locations from config
        location_descriptions = {
            "Lin Family Home": "A cozy family home where John, Mei, and Eddy Lin live",
            "Moreno Family Home": "Tom Moreno's family residence",
            "Moore Family Home": "The Moore family house where Sam lives",
            "The Willows": "An apartment complex with multiple units",
            "Oak Hill College": "Local college with classrooms and facilities",
            "Harvey Oak Supply Store": "Hardware store owned by Tom Moreno",
            "The Rose and Crown Pub": "Local pub for drinks and socializing",
            "Hobbs Cafe": "Coffee shop and cafe owned by Isabella Rodriguez",
            "Johnson Park": "Public park with recreational facilities",
            "Town Hall": "Municipal building for civic activities",
            "Library": "Public library with books and computers",
            "Pharmacy": "Pharmacy run by John Lin"
        }
        
        for location_name, sub_areas in SMALLVILLE_LOCATIONS.items():
            self.locations[location_name] = Location(
                name=location_name,
                sub_areas=sub_areas,
                description=location_descriptions.get(location_name, ""),
                current_agents=set()
            )
        
        # Add some objects to locations
        self._add_location_objects()
    
    def _add_location_objects(self):
        """Add objects to locations."""
        location_objects = {
            "Lin Family Home": ["dining table", "TV", "couch", "refrigerator", "bed"],
            "Moreno Family Home": ["dining table", "TV", "couch", "refrigerator", "bed"],
            "Moore Family Home": ["dining table", "TV", "couch", "refrigerator", "bed"],
            "The Willows": ["mailboxes", "elevator", "stairs"],
            "Oak Hill College": ["desks", "chalkboard", "computers", "books"],
            "Harvey Oak Supply Store": ["tools", "hardware", "paint", "lumber", "cash register"],
            "The Rose and Crown Pub": ["bar", "tables", "chairs", "dartboard", "TV"],
            "Hobbs Cafe": ["coffee machine", "tables", "chairs", "pastries", "menu board"],
            "Johnson Park": ["benches", "playground equipment", "walking path", "pond"],
            "Town Hall": ["meeting table", "podium", "filing cabinets", "flag"],
            "Library": ["books", "computers", "study tables", "reference desk"],
            "Pharmacy": ["medicine shelves", "prescription counter", "consultation room"]
        }
        
        for location_name, objects in location_objects.items():
            if location_name in self.locations:
                self.locations[location_name].objects = objects
    
    def _initialize_connections(self):
        """Initialize connections between locations (for travel time calculation)."""
        # Simple adjacency for Smallville - all locations are connected
        # In a more complex simulation, you'd have a proper map
        all_locations = list(self.locations.keys())
        for location in all_locations:
            # Connect each location to all others (small town)
            self.location_connections[location] = [
                loc for loc in all_locations if loc != location
            ]
    
    def move_agent(self, agent_name: str, destination: str, 
                   sub_area: Optional[str] = None) -> bool:
        """Move an agent to a new location."""
        if destination not in self.locations:
            from planning_utils import snap_to_valid_location
            # Try to snap hallucinated location to a valid one
            home = self.agent_locations.get(agent_name, "Oak Hill College")
            snapped = snap_to_valid_location(destination, default=home)
            if snapped in self.locations:
                logger.info(f"Snapped location '{destination}' → '{snapped}' for {agent_name}")
                destination = snapped
            else:
                logger.warning(f"Unknown location: {destination}")
                return False
        
        # Remove from current location
        current_location = self.agent_locations.get(agent_name)
        if current_location and current_location in self.locations:
            self.locations[current_location].current_agents.discard(agent_name)
        
        # Add to new location
        location = self.locations[destination]
        if len(location.current_agents) >= location.capacity:
            logger.warning(f"Location {destination} is at capacity")
            return False
        
        location.current_agents.add(agent_name)
        self.agent_locations[agent_name] = destination
        
        # Set sub-area if specified
        if sub_area and sub_area in location.sub_areas:
            self.agent_sub_areas[agent_name] = sub_area
        elif location.sub_areas:
            # Default to first sub-area
            self.agent_sub_areas[agent_name] = location.sub_areas[0]
        else:
            self.agent_sub_areas[agent_name] = ""
        
        logger.info(f"{agent_name} moved to {destination}" + 
                   (f" ({sub_area})" if sub_area else ""))
        return True
    
    def get_agent_location(self, agent_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Get an agent's current location and sub-area."""
        location = self.agent_locations.get(agent_name)
        sub_area = self.agent_sub_areas.get(agent_name)
        return location, sub_area
    
    def get_agents_at_location(self, location_name: str, 
                              sub_area: Optional[str] = None) -> List[str]:
        """Get all agents at a specific location (and optionally sub-area)."""
        if location_name not in self.locations:
            return []
        
        agents = list(self.locations[location_name].current_agents)
        
        if sub_area:
            # Filter by sub-area
            agents = [
                agent for agent in agents 
                if self.agent_sub_areas.get(agent) == sub_area
            ]
        
        return agents
    
    def get_nearby_agents(self, agent_name: str) -> List[str]:
        """Get agents in the same location and sub-area."""
        location, sub_area = self.get_agent_location(agent_name)
        if not location:
            return []
        
        nearby_agents = self.get_agents_at_location(location, sub_area)
        # Remove the agent themselves
        return [agent for agent in nearby_agents if agent != agent_name]
    
    def calculate_travel_time(self, origin: str, destination: str) -> int:
        """Calculate travel time between locations in minutes."""
        if origin == destination:
            return 0
        
        # Simple travel time model for Smallville
        # All locations are within walking distance
        travel_times = {
            ("Lin Family Home", "Pharmacy"): 2,
            ("Moreno Family Home", "Harvey Oak Supply Store"): 2,
            ("Oak Hill College", "Library"): 3,
            ("Hobbs Cafe", "The Rose and Crown Pub"): 4,
        }
        
        # Check direct mapping
        key = (origin, destination)
        if key in travel_times:
            return travel_times[key]
        
        # Check reverse mapping
        reverse_key = (destination, origin)
        if reverse_key in travel_times:
            return travel_times[reverse_key]
        
        # Default travel times based on location types
        home_locations = ["Lin Family Home", "Moreno Family Home", "Moore Family Home", "The Willows"]
        business_locations = ["Harvey Oak Supply Store", "Hobbs Cafe", "The Rose and Crown Pub", "Pharmacy"]
        public_locations = ["Oak Hill College", "Johnson Park", "Town Hall", "Library"]
        
        def get_location_type(loc):
            if loc in home_locations:
                return "home"
            elif loc in business_locations:
                return "business"
            elif loc in public_locations:
                return "public"
            return "other"
        
        origin_type = get_location_type(origin)
        dest_type = get_location_type(destination)
        
        # Default travel times (minutes)
        if origin_type == dest_type:
            return 5  # Same area type
        elif "home" in [origin_type, dest_type]:
            return 7  # From/to residential
        else:
            return 6  # Between different business/public areas
    
    def get_location_description(self, location_name: str, 
                               include_agents: bool = True, 
                               include_objects: bool = True) -> str:
        """Get a description of a location."""
        if location_name not in self.locations:
            return f"Unknown location: {location_name}"
        
        location = self.locations[location_name]
        description = f"{location.name}: {location.description}"
        
        if include_objects and location.objects:
            objects_str = ", ".join(location.objects[:5])  # Limit to 5 objects
            description += f"\nObjects here: {objects_str}"
        
        if include_agents and location.current_agents:
            agents_str = ", ".join(sorted(location.current_agents))
            description += f"\nPeople here: {agents_str}"
        
        if location.sub_areas:
            sub_areas_str = ", ".join(location.sub_areas)
            description += f"\nAreas: {sub_areas_str}"
        
        return description
    
    def get_available_locations(self) -> List[str]:
        """Get list of all available locations."""
        return list(self.locations.keys())
    
    def get_sub_areas(self, location_name: str) -> List[str]:
        """Get sub-areas for a location."""
        if location_name not in self.locations:
            return []
        return self.locations[location_name].sub_areas.copy()
    
    def observe_environment(self, agent_name: str) -> List[str]:
        """Generate observations for an agent based on their environment."""
        observations = []
        location, sub_area = self.get_agent_location(agent_name)
        
        if not location:
            return observations
        
        # Observe other agents
        nearby_agents = self.get_nearby_agents(agent_name)
        if nearby_agents:
            for other_agent in nearby_agents:
                observations.append(f"{other_agent} is in {location}")
        
        # Observe objects (occasionally)
        location_obj = self.locations[location]
        if location_obj.objects and len(observations) < 2:  # Don't overwhelm with objects
            import random
            if random.random() < 0.3:  # 30% chance to notice objects
                obj = random.choice(location_obj.objects)
                observations.append(f"There is a {obj} in {location}")
        
        return observations
    
    def get_environment_state(self) -> Dict:
        """Get the current state of the environment."""
        state = {
            "locations": {},
            "agent_locations": self.agent_locations.copy(),
            "agent_sub_areas": self.agent_sub_areas.copy()
        }
        
        for name, location in self.locations.items():
            state["locations"][name] = {
                "current_agents": list(location.current_agents),
                "sub_areas": location.sub_areas,
                "objects": location.objects,
                "description": location.description
            }
        
        return state
    
    def load_environment_state(self, state: Dict):
        """Load environment state."""
        self.agent_locations = state.get("agent_locations", {})
        self.agent_sub_areas = state.get("agent_sub_areas", {})
        
        # Update location agent sets
        for location in self.locations.values():
            location.current_agents.clear()
        
        for agent, location_name in self.agent_locations.items():
            if location_name in self.locations:
                self.locations[location_name].current_agents.add(agent)