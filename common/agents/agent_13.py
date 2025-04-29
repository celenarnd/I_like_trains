import logging
import math
from typing import Tuple, List, Dict, Any, Optional
import heapq

from common.base_agent import BaseAgent
from common.move import Move

class Agent(BaseAgent):
    """
    Passenger Magnet Agent
    
    Identifies and moves toward the location with the highest number of passengers.
    Tries to load as many passengers as possible before heading to drop them off.
    Prioritizes quantity over speed.
    Uses shortest pathfinding and avoids unnecessary detours.
    """
    
    def __init__(self, nickname, network, logger="client.agent"):
        super().__init__(nickname, network, logger)
        self.target_location = None
        self.pickup_threshold = 5  # Higher threshold - want to collect more passengers
        self.going_to_delivery = False
        self.path = []  # Path to current target
        self.last_target_value = 0  # Track the value of our current target
        self.stuck_counter = 0  # Counter to detect if we're stuck
        self.last_position = None  # Track our last position
    
    def get_move(self) -> Move:
        """Return the next move for the passenger magnet agent."""
        # Make sure we have all the needed game state
        if not all([self.all_trains, self.passengers, self.delivery_zone, 
                   self.game_width, self.game_height, self.cell_size]):
            return Move.RIGHT  # Default move if we don't have all the info
            
        # Get our train details
        if self.nickname not in self.all_trains:
            return Move.RIGHT  # Default move if we can't find our train
            
        train = self.all_trains[self.nickname]
        current_pos = train["position"]
        current_direction = train["direction"]
        wagon_count = len(train.get("wagons", []))
        
        # Check if we're stuck
        if self.last_position == current_pos:
            self.stuck_counter += 1
            if self.stuck_counter > 5:  # If stuck for more than 5 moves
                self.path = []  # Clear path
                self.stuck_counter = 0
        else:
            self.stuck_counter = 0
        
        # Update last position
        self.last_position = current_pos
        
        # If we're in the delivery zone and have wagons, drop them
        if self.is_in_delivery_zone(current_pos) and wagon_count > 0:
            self.going_to_delivery = False
            return Move.DROP
            
        # If we have enough passengers or if we've been collecting for a while,
        # head to delivery zone
        if (wagon_count >= self.pickup_threshold and not self.going_to_delivery) or \
           (wagon_count > 0 and not self.going_to_delivery and self.should_deliver_early(current_pos)):
            self.going_to_delivery = True
            self.target_location = None
            self.path = []
        
        # If we're heading to delivery zone, calculate path to delivery zone
        if self.going_to_delivery:
            return self.move_towards_delivery(current_pos, current_direction)
            
        # Otherwise, go for the highest value passenger hotspot
        return self.move_towards_hotspot(current_pos, current_direction, wagon_count)
    
    def should_deliver_early(self, current_pos: Tuple[int, int]) -> bool:
        """Decide if we should deliver early based on distance to delivery zone and passenger availability."""
        if not self.passengers:
            return True  # No more passengers, might as well deliver
            
        # Calculate distance to delivery zone
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        dz_center = (dz_pos[0] + dz_width // 2, dz_pos[1] + dz_height // 2)
        
        distance_to_delivery = self.manhattan_distance(current_pos, dz_center)
        
        # If we're close to delivery zone, go ahead and deliver
        if distance_to_delivery < 5 * self.cell_size:
            return True
            
        # If we're far from delivery zone and there are few passengers, deliver
        if len(self.passengers) < 3:
            return True
            
        return False
    
    def find_passenger_hotspots(self) -> List[Dict]:
        """Identify passenger hotspots by grouping nearby passengers."""
        if not self.passengers:
            return []
            
        # Start with each passenger as its own hotspot
        hotspots = []
        for passenger in self.passengers:
            pos = passenger["position"]
            value = passenger["value"]
            hotspots.append({
                "position": pos,
                "value": value,
                "count": 1
            })
        
        # Merge hotspots that are close to each other
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(hotspots):
                j = i + 1
                while j < len(hotspots):
                    if self.manhattan_distance(hotspots[i]["position"], hotspots[j]["position"]) <= 3 * self.cell_size:
                        # Merge these hotspots
                        hotspots[i]["value"] += hotspots[j]["value"]
                        hotspots[i]["count"] += hotspots[j]["count"]
                        # Update position to be between the two
                        pos1 = hotspots[i]["position"]
                        pos2 = hotspots[j]["position"]
                        hotspots[i]["position"] = (
                            (pos1[0] + pos2[0]) // 2,
                            (pos1[1] + pos2[1]) // 2
                        )
                        # Remove the merged hotspot
                        hotspots.pop(j)
                        merged = True
                    else:
                        j += 1
                i += 1
        
        return hotspots
    
    def move_towards_hotspot(self, current_pos: Tuple[int, int], 
                           current_direction: Tuple[int, int],
                           wagon_count: int) -> Move:
        """Move towards the highest value passenger hotspot."""
        # If no passengers, just continue in current direction
        if not self.passengers:
            return Move(current_direction)
            
        # Find passenger hotspots
        hotspots = self.find_passenger_hotspots()
        
        # Score hotspots based on value and distance
        scored_hotspots = []
        for hotspot in hotspots:
            pos = hotspot["position"]
            value = hotspot["value"]
            count = hotspot["count"]
            distance = self.manhattan_distance(current_pos, pos)
            
            # Higher value, higher count, and shorter distance is better
            # Use a value multiplier that increases with wagon count
            value_multiplier = 1.0 + (count * 0.5)
            # Use a distance penalty that decreases with wagon count (reduced mobility)
            distance_penalty = max(1.0, 1.0 + (wagon_count * 0.1))
            
            score = (value * value_multiplier) / max(1, (distance / self.cell_size) * distance_penalty)
            scored_hotspots.append((hotspot, score))
        
        # Sort by score (higher is better)
        scored_hotspots.sort(key=lambda x: x[1], reverse=True)
        
        # Check if we should change target
        if self.target_location:
            # If current target is still a valid hotspot, keep it
            current_target_valid = False
            current_target_value = 0
            for hotspot, _ in scored_hotspots:
                if hotspot["position"] == self.target_location:
                    current_target_valid = True
                    current_target_value = hotspot["value"]
                    break
            
            # If our target is gone or significantly less valuable, choose a new one
            if not current_target_valid or current_target_value < self.last_target_value * 0.5:
                self.target_location = scored_hotspots[0][0]["position"]
                self.last_target_value = scored_hotspots[0][0]["value"]
                self.path = []
        else:
            # No current target, choose the best hotspot
            self.target_location = scored_hotspots[0][0]["position"]
            self.last_target_value = scored_hotspots[0][0]["value"]
        
        # Calculate path to the hotspot
        return self.get_next_move_in_path(current_pos, self.target_location, current_direction)
    
    def move_towards_delivery(self, current_pos: Tuple[int, int], 
                            current_direction: Tuple[int, int]) -> Move:
        """Move towards the delivery zone using shortest path."""
        # Calculate the center of the delivery zone
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        
        # Target the center of the delivery zone
        target_pos = (
            dz_pos[0] + dz_width // 2,
            dz_pos[1] + dz_height // 2
        )
        
        # Get next move in path to delivery zone
        return self.get_next_move_in_path(current_pos, target_pos, current_direction)
    
    def get_next_move_in_path(self, current_pos: Tuple[int, int], 
                             target_pos: Tuple[int, int], 
                             current_direction: Tuple[int, int]) -> Move:
        """Calculate the next move in the path using A* algorithm."""
        # If we're already at the target, no need to move
        if current_pos == target_pos:
            return Move(current_direction)
        
        # If we already have a path, follow it
        if self.path:
            next_pos = self.path[0]
            dx = (next_pos[0] - current_pos[0]) // self.cell_size
            dy = (next_pos[1] - current_pos[1]) // self.cell_size
            
            # If we've reached this position, remove it from the path
            if current_pos == next_pos:
                self.path.pop(0)
                return self.get_next_move_in_path(current_pos, target_pos, current_direction)
            
            # Convert the direction to a Move
            for move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
                if move.value == (dx, dy):
                    # Can't make U-turns
                    if not self.is_opposite_direction(move.value, current_direction):
                        return move
            
            # If we can't make the move, recalculate the path
            self.path = []
        
        # Calculate a new path using A*
        self.path = self.find_path(current_pos, target_pos, current_direction)
        
        # If no path was found, use simple directional movement
        if not self.path:
            return self.direct_move_towards(current_pos, target_pos, current_direction)
        
        # Return the next move in the path
        return self.get_next_move_in_path(current_pos, target_pos, current_direction)
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 current_direction: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a path from start to goal using A* algorithm."""
        # Initialize open and closed sets
        open_set = []
        closed_set = set()
        
        # Start node
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}
        came_from = {}
        
        # Add start node to open set
        heapq.heappush(open_set, (f_score[start], start))
        
        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)
            
            # If we've reached the goal, reconstruct the path
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            # Add current to closed set
            closed_set.add(current)
            
            # Get neighbors
            neighbors = self.get_valid_neighbors(current, current_direction if current == start else None)
            
            for neighbor in neighbors:
                # Skip if neighbor is already processed
                if neighbor in closed_set:
                    continue
                
                # Calculate g_score for this neighbor
                tentative_g_score = g_score[current] + self.cell_size
                
                # If this is a new node or we found a better path
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update scores
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.manhattan_distance(neighbor, goal)
                    
                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return []
    
    def get_valid_neighbors(self, pos: Tuple[int, int], 
                          current_direction: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        neighbors = []
        
        for move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
            # Skip if this would be a U-turn
            if current_direction and self.is_opposite_direction(move.value, current_direction):
                continue
                
            # Calculate new position
            new_x = pos[0] + move.value[0] * self.cell_size
            new_y = pos[1] + move.value[1] * self.cell_size
            new_pos = (new_x, new_y)
            
            # Check if valid (within bounds)
            if self.is_valid_position(new_pos):
                # Check if there's a train or wagon at this position
                if self.is_train_at_position(new_pos):
                    continue
                
                neighbors.append(new_pos)
                
        return neighbors
    
    def is_train_at_position(self, pos: Tuple[int, int]) -> bool:
        """Check if there's a train or wagon at a given position."""
        for name, train in self.all_trains.items():
            if pos == train["position"]:
                return True
                
            for wagon_pos in train.get("wagons", []):
                if pos == wagon_pos:
                    return True
                    
        return False
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within bounds)."""
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= self.game_width or y < 0 or y >= self.game_height:
            return False
            
        return True
    
    def direct_move_towards(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                           current_direction: Tuple[int, int]) -> Move:
        """Make a direct move towards the target position."""
        # Calculate the direction to the target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Get all possible moves (excluding U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        valid_moves = [move for move in possible_moves 
                      if not self.is_opposite_direction(move.value, current_direction)]
        
        # Check if we're aligned with the target on either axis
        if dx == 0 and any(move.value[0] == 0 for move in valid_moves):
            # We're aligned on x-axis, try to move on y-axis
            if dy < 0 and Move.UP in valid_moves:
                return Move.UP
            elif dy > 0 and Move.DOWN in valid_moves:
                return Move.DOWN
                
        if dy == 0 and any(move.value[1] == 0 for move in valid_moves):
            # We're aligned on y-axis, try to move on x-axis
            if dx < 0 and Move.LEFT in valid_moves:
                return Move.LEFT
            elif dx > 0 and Move.RIGHT in valid_moves:
                return Move.RIGHT
        
        # Not aligned on either axis, calculate the best move
        best_move = min(valid_moves, 
                       key=lambda move: self.get_distance_after_move(current_pos, move.value, target_pos))
        
        return best_move
    
    def get_distance_after_move(self, current_pos: Tuple[int, int], 
                               move_dir: Tuple[int, int], 
                               target_pos: Tuple[int, int]) -> float:
        """Calculate the distance to the target after making a move."""
        new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                  current_pos[1] + move_dir[1] * self.cell_size)
        return self.manhattan_distance(new_pos, target_pos)
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_opposite_direction(self, dir1: Tuple[int, int], dir2: Tuple[int, int]) -> bool:
        """Check if two directions are opposite."""
        return dir1[0] == -dir2[0] and dir1[1] == -dir2[1]
    
    def is_in_delivery_zone(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside the delivery zone."""
        x, y = position
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        
        return (x >= dz_pos[0] and x < dz_pos[0] + dz_width and
                y >= dz_pos[1] and y < dz_pos[1] + dz_height) 