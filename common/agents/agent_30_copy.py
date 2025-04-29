import logging
import math
import heapq
from typing import Tuple, List, Dict, Any, Optional, Set

from common.base_agent import BaseAgent
from common.move import Move

class Agent(BaseAgent):
    """
    Pathfinding Agent
    
    Uses A* algorithm to plan optimal paths for picking up and delivering passengers.
    Dynamically updates paths to avoid collisions and minimize risks.
    Balances efficiency with safety by considering danger zones in pathfinding.
    Uses diagonal movement by alternating between horizontal and vertical moves.
    """
    
    def __init__(self, nickname, network, logger="client.agent"):
        super().__init__(nickname, network, logger)
        self.grid_size = None  # Size of the grid (width, height)
        self.grid = None  # Grid representation of the game
        self.path = []  # Current path being followed
        self.target = None  # Current target (passenger or delivery point)
        self.target_type = None  # "passenger" or "delivery"
        self.pickup_threshold = 3  # Collect this many passengers before delivery
        self.path_recalculation_frequency = 5  # Recalculate path every N frames
        self.frame_counter = 0  # Counter for path recalculation
        self.last_move = None  # Track last move for diagonal movement
        self.danger_lookahead = 5  # Look 5 cells ahead for danger zones (improved from 3)
        self.risk_threshold = 0.7  # Threshold for considering a move too risky
        
    def get_move(self) -> Move:
        """Return the next move for the pathfinding agent."""
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
        
        # Update our grid representation of the game
        self.update_grid()
        
        # Increment frame counter for periodic path recalculation
        self.frame_counter = (self.frame_counter + 1) % self.path_recalculation_frequency
        
        # If we're in the delivery zone and have wagons, drop them
        if self.is_in_delivery_zone(current_pos) and wagon_count > 0:
            self.target = None
            self.target_type = None
            return Move.DROP
        
        # Check if we need to recalculate our path
        if (self.needs_new_path(current_pos, wagon_count) or 
            self.frame_counter == 0 or  # Periodic recalculation
            (len(self.path) > 0 and not self.is_safe_path())):
            
            # Determine whether to target a passenger or the delivery zone
            if wagon_count >= self.pickup_threshold:
                self.target_type = "delivery"
                delivery_point = self.find_best_delivery_point(current_pos)
                self.target = delivery_point
                self.path = self.find_path_astar(current_pos, delivery_point, current_direction)
            else:
                self.target_type = "passenger"
                best_passenger = self.find_best_passenger(current_pos)
                if best_passenger:
                    self.target = best_passenger["position"]
                    self.path = self.find_path_astar(current_pos, self.target, current_direction)
                else:
                    self.path = []  # No passengers available
        
        # Check if any passenger is within 3 cells - always try to pick them up regardless of wagon count
        nearby_passenger = self.check_nearby_passenger(current_pos, current_direction)
        if nearby_passenger:
            return nearby_passenger
            
        # If we have a path, follow it with diagonal movement if possible
        if self.path:
            # Get the next position in the path
            next_pos = self.path[0]
            
            # Try diagonal movement first
            diagonal_move = self.move_diagonally(current_pos, next_pos, current_direction)
            if diagonal_move:
                return diagonal_move
                
            # If diagonal movement not possible, use direct movement
            direct_move = self.get_move_to_adjacent_cell(current_pos, next_pos, current_direction)
            if direct_move:
                self.path.pop(0)  # Remove the cell we're moving to
                self.last_move = direct_move
                return direct_move
        
        # If we have no path or can't follow it, use a default safe move
        return self.get_safe_default_move(current_pos, current_direction)
    
    def update_grid(self):
        """Update the grid representation of the game."""
        # Calculate grid size based on cell_size
        grid_width = self.game_width // self.cell_size
        grid_height = self.game_height // self.cell_size
        
        if self.grid_size != (grid_width, grid_height):
            self.grid_size = (grid_width, grid_height)
            self.grid = [[0 for _ in range(grid_height)] for _ in range(grid_width)]
        
        # Reset the grid (0 = empty, 1 = obstacle, 2-7 = danger zones with different levels)
        for x in range(grid_width):
            for y in range(grid_height):
                self.grid[x][y] = 0
        
        # Mark obstacles (trains and wagons)
        for train_name, train_data in self.all_trains.items():
            # Mark the train head position
            train_pos = train_data["position"]
            grid_x = train_pos[0] // self.cell_size
            grid_y = train_pos[1] // self.cell_size
            
            # Ensure we're within grid bounds
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                self.grid[grid_x][grid_y] = 1  # Mark as obstacle
            
            # Mark wagon positions as obstacles
            for wagon_pos in train_data.get("wagons", []):
                grid_x = wagon_pos[0] // self.cell_size
                grid_y = wagon_pos[1] // self.cell_size
                
                # Ensure we're within grid bounds
                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    self.grid[grid_x][grid_y] = 1  # Mark as obstacle
            
            # Mark predicted future positions of other trains as danger zones
            if train_name != self.nickname:
                train_dir = train_data["direction"]
                
                # Mark cells in front of other trains as dangerous - now looking 5 cells ahead
                for i in range(1, self.danger_lookahead + 1):
                    future_x = train_pos[0] + train_dir[0] * self.cell_size * i
                    future_y = train_pos[1] + train_dir[1] * self.cell_size * i
                    
                    grid_x = future_x // self.cell_size
                    grid_y = future_y // self.cell_size
                    
                    # Ensure we're within grid bounds
                    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                        if self.grid[grid_x][grid_y] == 0:  # Don't overwrite obstacles
                            # Add decreasing danger levels based on distance
                            danger_level = 2 + (self.danger_lookahead - i)  # Higher values for closer dangers
                            self.grid[grid_x][grid_y] = danger_level
                            
                # Also mark cells adjacent to the train's path as mild danger zones
                for i in range(1, self.danger_lookahead):
                    future_x = train_pos[0] + train_dir[0] * self.cell_size * i
                    future_y = train_pos[1] + train_dir[1] * self.cell_size * i
                    
                    # Mark adjacent cells (perpendicular to travel direction)
                    if train_dir[0] == 0:  # Moving vertically
                        for dx in [-1, 1]:
                            adj_x = future_x + dx * self.cell_size
                            adj_y = future_y
                            
                            grid_x = adj_x // self.cell_size
                            grid_y = adj_y // self.cell_size
                            
                            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                                if self.grid[grid_x][grid_y] == 0:  # Don't overwrite obstacles or higher dangers
                                    self.grid[grid_x][grid_y] = 2  # Mild danger
                    else:  # Moving horizontally
                        for dy in [-1, 1]:
                            adj_x = future_x
                            adj_y = future_y + dy * self.cell_size
                            
                            grid_x = adj_x // self.cell_size
                            grid_y = adj_y // self.cell_size
                            
                            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                                if self.grid[grid_x][grid_y] == 0:  # Don't overwrite obstacles or higher dangers
                                    self.grid[grid_x][grid_y] = 2  # Mild danger
    
    def needs_new_path(self, current_pos: Tuple[int, int], wagon_count: int) -> bool:
        """Decide if we need to recalculate the path."""
        # If we have no path, we need a new one
        if not self.path:
            return True
        
        # If we've just collected enough passengers to deliver
        if wagon_count >= self.pickup_threshold and self.target_type == "passenger":
            return True
        
        # If we've just delivered all passengers
        if wagon_count == 0 and self.target_type == "delivery":
            return True
        
        # If our target is a passenger, check if it's still there
        if self.target_type == "passenger":
            passenger_positions = [p["position"] for p in self.passengers]
            if self.target not in passenger_positions:
                return True
        
        # If we find a much better passenger when we're still collecting
        if self.target_type == "passenger" and wagon_count < self.pickup_threshold:
            current_target_value = 0
            for p in self.passengers:
                if p["position"] == self.target:
                    current_target_value = p["value"]
                    break
            
            # Check for a significantly better passenger nearby
            for p in self.passengers:
                if p["position"] != self.target and p["value"] > current_target_value + 2:
                    # If it's nearby and much more valuable
                    distance = self.get_distance(current_pos, p["position"])
                    if distance < self.cell_size * 5:  # Within 5 cells
                        return True
        
        return False
    
    def is_safe_path(self) -> bool:
        """Check if the current path is still safe (no new obstacles or dangers)."""
        if not self.path:
            return False
        
        for pos in self.path:
            grid_x = pos[0] // self.cell_size
            grid_y = pos[1] // self.cell_size
            
            # Check if position is within grid bounds
            if (0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]):
                # Check if the cell is now an obstacle
                if self.grid[grid_x][grid_y] == 1:
                    return False
        
        return True
    
    def find_best_passenger(self, current_pos: Tuple[int, int]) -> Optional[Dict]:
        """Find the best passenger to target based on value, distance, and hotspots."""
        if not self.passengers:
            return None
        
        # First identify passenger hotspots by grouping nearby passengers
        hotspots = self.find_passenger_hotspots()
        
        # Score each hotspot based on value, distance, and safety
        scored_hotspots = []
        for hotspot in hotspots:
            hotspot_pos = hotspot["position"]
            hotspot_value = hotspot["value"]
            hotspot_count = hotspot["count"]
            
            # Check if the hotspot's position is an obstacle or danger
            grid_x = hotspot_pos[0] // self.cell_size
            grid_y = hotspot_pos[1] // self.cell_size
            
            # Skip hotspots at obstacle positions
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1] and 
                self.grid[grid_x][grid_y] == 1):
                continue
            
            # Calculate distance to hotspot
            distance = self.get_distance(current_pos, hotspot_pos)
            
            # Calculate risk level at the hotspot
            risk_level = self.assess_risk(hotspot_pos)
            
            # Value multiplier that increases with passenger count
            value_multiplier = 1.0 + (hotspot_count * 0.5)
            
            # Safety penalty that increases with risk level
            safety_penalty = 1.0 + (risk_level * 2.0)
            
            # Score formula: balance of value, count, distance, and safety
            # Higher score is better
            score = (hotspot_value * value_multiplier) / max(1, (distance / self.cell_size) * safety_penalty)
            scored_hotspots.append((hotspot, score))
        
        # If no valid hotspots, return None
        if not scored_hotspots:
            return None
        
        # Sort by score (higher is better)
        scored_hotspots.sort(key=lambda x: x[1], reverse=True)
        
        # Find the best passenger in the best hotspot
        best_hotspot = scored_hotspots[0][0]
        best_passenger = None
        best_value = -1
        
        for passenger in self.passengers:
            passenger_pos = passenger["position"]
            # Check if this passenger is in or near the best hotspot
            if self.get_distance(passenger_pos, best_hotspot["position"]) <= 3 * self.cell_size:
                # Consider both value and risk when selecting the best passenger
                passenger_risk = self.assess_risk(passenger_pos)
                # Only select passengers with acceptable risk
                if passenger_risk < self.risk_threshold:
                    passenger_score = passenger["value"] * (1.0 - passenger_risk)
                    if passenger_score > best_value:
                        best_value = passenger_score
                        best_passenger = passenger
        
        # If we found a good passenger in the hotspot, return it
        if best_passenger:
            return best_passenger
        
        # Fallback to evaluating individual passengers with risk assessment
        scored_individual_passengers = []
        for passenger in self.passengers:
            pos = passenger["position"]
            value = passenger["value"]
            risk = self.assess_risk(pos)
            
            # Skip very high-risk passengers
            if risk >= self.risk_threshold:
                continue
                
            distance = self.get_distance(current_pos, pos)
            # Balance value, distance, and risk
            score = (value * (1.0 - risk)) / max(1, distance / self.cell_size)
            scored_individual_passengers.append((passenger, score))
            
        if scored_individual_passengers:
            scored_individual_passengers.sort(key=lambda x: x[1], reverse=True)
            return scored_individual_passengers[0][0]
            
        # If all passengers are too risky, return the highest value one as a last resort
        return max(self.passengers, key=lambda p: p["value"])
    
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
                    if self.get_distance(hotspots[i]["position"], hotspots[j]["position"]) <= 3 * self.cell_size:
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
        
    def assess_risk(self, position: Tuple[int, int]) -> float:
        """Assess the risk level of a position (0.0 = safe, 1.0 = dangerous)."""
        # Convert to grid coordinates
        grid_x = position[0] // self.cell_size
        grid_y = position[1] // self.cell_size
        
        # Check if position is within grid bounds
        if not (0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]):
            return 1.0  # Out of bounds is maximum risk
            
        # Check grid cell value
        grid_value = self.grid[grid_x][grid_y]
        
        # If it's an obstacle, maximum risk
        if grid_value == 1:
            return 1.0
            
        # If it's a danger zone, scale the risk based on danger level
        if grid_value > 1:
            # Convert danger level (2-7) to risk (0.3-0.9)
            max_danger = 1 + self.danger_lookahead
            normalized_danger = (grid_value - 1) / max_danger
            return 0.3 + (normalized_danger * 0.6)
            
        # Calculate risk based on proximity to other trains
        risk = 0.0
        for train_name, train_data in self.all_trains.items():
            if train_name == self.nickname:
                continue
                
            train_pos = train_data["position"]
            train_dir = train_data["direction"]
            
            # Calculate distance to train
            distance = self.get_distance(position, train_pos)
            
            # Higher risk for closer trains
            if distance < self.cell_size * self.danger_lookahead:
                # Risk decreases with distance
                proximity_risk = 1.0 - (distance / (self.cell_size * self.danger_lookahead))
                
                # Check if we're in the train's path
                in_path = False
                
                # For horizontal trains
                if train_dir[0] != 0 and train_dir[1] == 0:
                    # Check if we're in the same row and in front of the train
                    same_row = abs(position[1] - train_pos[1]) < self.cell_size
                    in_front = (position[0] - train_pos[0]) * train_dir[0] > 0
                    if same_row and in_front:
                        in_path = True
                        
                # For vertical trains
                elif train_dir[0] == 0 and train_dir[1] != 0:
                    # Check if we're in the same column and in front of the train
                    same_col = abs(position[0] - train_pos[0]) < self.cell_size
                    in_front = (position[1] - train_pos[1]) * train_dir[1] > 0
                    if same_col and in_front:
                        in_path = True
                
                # Higher risk if in train's path
                path_multiplier = 2.0 if in_path else 1.0
                
                # Update overall risk
                risk = max(risk, proximity_risk * path_multiplier)
        
        return min(1.0, risk)  # Cap risk at 1.0
    
    def find_best_delivery_point(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Find the best point in the delivery zone to target."""
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        
        # Check all edge points of the delivery zone
        best_point = None
        best_score = float('-inf')
        
        # Check horizontal edges
        for x in range(dz_pos[0], dz_pos[0] + dz_width, self.cell_size):
            # Top edge
            y = dz_pos[1]
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            
            # Skip points that are obstacles
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1] and 
                self.grid[grid_x][grid_y] != 1):
                
                # Calculate score based on distance and danger
                distance = self.get_distance(current_pos, (x, y))
                danger = self.grid[grid_x][grid_y] == 2  # Is it a danger zone?
                score = -distance - (5 * self.cell_size if danger else 0)
                
                if score > best_score:
                    best_score = score
                    best_point = (x, y)
            
            # Bottom edge
            y = dz_pos[1] + dz_height - self.cell_size
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            
            # Skip points that are obstacles
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1] and 
                self.grid[grid_x][grid_y] != 1):
                
                # Calculate score based on distance and danger
                distance = self.get_distance(current_pos, (x, y))
                danger = self.grid[grid_x][grid_y] == 2  # Is it a danger zone?
                score = -distance - (5 * self.cell_size if danger else 0)
                
                if score > best_score:
                    best_score = score
                    best_point = (x, y)
        
        # Check vertical edges
        for y in range(dz_pos[1], dz_pos[1] + dz_height, self.cell_size):
            # Left edge
            x = dz_pos[0]
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            
            # Skip points that are obstacles
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1] and 
                self.grid[grid_x][grid_y] != 1):
                
                # Calculate score based on distance and danger
                distance = self.get_distance(current_pos, (x, y))
                danger = self.grid[grid_x][grid_y] == 2  # Is it a danger zone?
                score = -distance - (5 * self.cell_size if danger else 0)
                
                if score > best_score:
                    best_score = score
                    best_point = (x, y)
            
            # Right edge
            x = dz_pos[0] + dz_width - self.cell_size
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            
            # Skip points that are obstacles
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1] and 
                self.grid[grid_x][grid_y] != 1):
                
                # Calculate score based on distance and danger
                distance = self.get_distance(current_pos, (x, y))
                danger = self.grid[grid_x][grid_y] == 2  # Is it a danger zone?
                score = -distance - (5 * self.cell_size if danger else 0)
                
                if score > best_score:
                    best_score = score
                    best_point = (x, y)
        
        # If no suitable point found, use center of delivery zone
        if not best_point:
            return (dz_pos[0] + dz_width // 2, dz_pos[1] + dz_height // 2)
        
        return best_point 
    
    def find_path_astar(self, start: Tuple[int, int], goal: Tuple[int, int],
                       current_direction: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a path using A* algorithm with enhanced risk awareness."""
        # Convert positions to grid coordinates
        start_grid = (start[0] // self.cell_size, start[1] // self.cell_size)
        goal_grid = (goal[0] // self.cell_size, goal[1] // self.cell_size)
        
        # Check if goal is an obstacle
        if (0 <= goal_grid[0] < self.grid_size[0] and 
            0 <= goal_grid[1] < self.grid_size[1] and 
            self.grid[goal_grid[0]][goal_grid[1]] == 1):
            
            # Find a nearby accessible cell if goal is blocked
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    
                    alt_x = goal_grid[0] + dx
                    alt_y = goal_grid[1] + dy
                    
                    if (0 <= alt_x < self.grid_size[0] and 
                        0 <= alt_y < self.grid_size[1] and 
                        self.grid[alt_x][alt_y] != 1):
                        
                        goal_grid = (alt_x, alt_y)
                        goal = (goal_grid[0] * self.cell_size, goal_grid[1] * self.cell_size)
                        break
                else:
                    continue
                break
        
        # Initialize the open and closed sets
        open_set = []
        closed_set = set()
        
        # Calculate the initial direction vector
        dir_vector = (
            1 if current_direction[0] > 0 else (-1 if current_direction[0] < 0 else 0),
            1 if current_direction[1] > 0 else (-1 if current_direction[1] < 0 else 0)
        )
        
        # Add the starting node to the open set
        # Format: (f_score, g_score, pos, parent_pos, direction)
        heapq.heappush(open_set, (0, 0, start_grid, None, dir_vector))
        
        # Create a dict to store the best g_score for each position
        g_scores = {start_grid: 0}
        
        # Create a dict to store the parent of each position
        parents = {}
        
        # The directions we can move in
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        
        while open_set:
            # Get the node with the lowest f_score
            _, g_score, current, parent, current_dir = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if current == goal_grid:
                # Reconstruct the path
                path = self.reconstruct_path(parents, current)
                # Convert grid coordinates back to pixel coordinates
                return [(x * self.cell_size, y * self.cell_size) for x, y in path]
            
            # Add the current node to the closed set
            closed_set.add(current)
            
            # Check each neighbor
            for direction in directions:
                # Cannot reverse direction
                if (direction[0] == -current_dir[0] and direction[1] == -current_dir[1]):
                    continue
                
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                
                # Check if the neighbor is valid (within grid and not an obstacle)
                if (neighbor[0] < 0 or neighbor[0] >= self.grid_size[0] or
                    neighbor[1] < 0 or neighbor[1] >= self.grid_size[1] or
                    self.grid[neighbor[0]][neighbor[1]] == 1):  # 1 = obstacle
                    continue
                
                # Check if the neighbor is in the closed set
                if neighbor in closed_set:
                    continue
                
                # Calculate the tentative g_score with advanced risk assessment
                # Add a turn penalty if we change direction
                # Add a danger penalty based on the risk level
                turn_penalty = 0
                danger_penalty = 0
                
                if direction != current_dir:
                    turn_penalty = 0.5  # Minor penalty for turning
                
                # Calculate risk at this position
                neighbor_pos = (neighbor[0] * self.cell_size, neighbor[1] * self.cell_size)
                risk_level = self.assess_risk(neighbor_pos)
                
                # Higher risk = higher penalty (exponential)
                if risk_level > 0:
                    # Exponential penalty for high-risk areas
                    danger_penalty = 1.0 * (risk_level ** 2) * 5.0
                
                tentative_g = g_score + 1.0 + turn_penalty + danger_penalty
                
                # Check if this path is better than any previous path to this neighbor
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    # Update the g_score
                    g_scores[neighbor] = tentative_g
                    
                    # Calculate the h_score (Manhattan distance to goal)
                    h_score = abs(neighbor[0] - goal_grid[0]) + abs(neighbor[1] - goal_grid[1])
                    
                    # Calculate the f_score
                    f_score = tentative_g + h_score
                    
                    # Add the neighbor to the open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, current, direction))
                    
                    # Update the parent
                    parents[neighbor] = current
        
        # If we get here, no path was found
        return []
    
    def reconstruct_path(self, parents, current):
        """Reconstruct the path from the goal to the start."""
        path = [current]
        while current in parents:
            current = parents[current]
            path.append(current)
        path.reverse()
        # Remove the first element (current position)
        if path:
            path.pop(0)
        return path
    
    def check_adjacent_passenger(self, current_pos: Tuple[int, int], 
                               current_direction: Tuple[int, int]) -> Optional[Move]:
        """Check if there's a passenger adjacent to us that we can pick up."""
        if not self.passengers:
            return None
        
        # Check each passenger
        for passenger in self.passengers:
            passenger_pos = passenger["position"]
            distance = self.get_distance(current_pos, passenger_pos)
            
            # If passenger is exactly one cell away
            if distance == self.cell_size:
                # Calculate the move to get to the passenger
                dx = passenger_pos[0] - current_pos[0]
                dy = passenger_pos[1] - current_pos[1]
                
                # Convert to a Move
                if dx > 0 and dy == 0:
                    move = Move.RIGHT
                elif dx < 0 and dy == 0:
                    move = Move.LEFT
                elif dx == 0 and dy > 0:
                    move = Move.DOWN
                elif dx == 0 and dy < 0:
                    move = Move.UP
                else:
                    continue
                
                # Check if this move would be a U-turn (which is not allowed)
                if move.value == (-current_direction[0], -current_direction[1]):
                    continue
                
                return move
        
        return None
    
    def move_diagonally(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                       current_direction: Tuple[int, int]) -> Optional[Move]:
        """Implement diagonal movement by alternating horizontal and vertical moves."""
        # Calculate the direction to the target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Only try diagonal movement if both x and y differences are non-zero
        if dx == 0 or dy == 0:
            return None
        
        # Normalize the direction
        dx = 1 if dx > 0 else -1
        dy = 1 if dy > 0 else -1
        
        # Convert to Move objects
        horizontal_move = Move.RIGHT if dx > 0 else Move.LEFT
        vertical_move = Move.DOWN if dy > 0 else Move.UP
        
        # Check if these moves would be U-turns
        horizontal_is_uturn = horizontal_move.value == (-current_direction[0], -current_direction[1])
        vertical_is_uturn = vertical_move.value == (-current_direction[0], -current_direction[1])
        
        # If both are U-turns, we can't move diagonally
        if horizontal_is_uturn and vertical_is_uturn:
            return None
        
        # If one is a U-turn, use the other one
        if horizontal_is_uturn:
            self.last_move = vertical_move
            return vertical_move
        if vertical_is_uturn:
            self.last_move = horizontal_move
            return horizontal_move
        
        # For diagonal movement, alternate between horizontal and vertical
        if self.last_move:
            if self.last_move in [Move.LEFT, Move.RIGHT]:
                # Last move was horizontal, try vertical next
                next_move = vertical_move
            else:
                # Last move was vertical, try horizontal next
                next_move = horizontal_move
        else:
            # No last move, start with horizontal
            next_move = horizontal_move
        
        self.last_move = next_move
        return next_move
    
    def get_move_to_adjacent_cell(self, current_pos: Tuple[int, int], 
                                target_pos: Tuple[int, int], 
                                current_direction: Tuple[int, int]) -> Optional[Move]:
        """Get the move to go from current position to an adjacent cell."""
        # Calculate the direction vector
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Normalize the direction to match cell size
        if dx != 0:
            dx = dx // abs(dx) * self.cell_size
        if dy != 0:
            dy = dy // abs(dy) * self.cell_size
        
        # Convert to a Move
        if dx > 0 and dy == 0:
            move = Move.RIGHT
        elif dx < 0 and dy == 0:
            move = Move.LEFT
        elif dx == 0 and dy > 0:
            move = Move.DOWN
        elif dx == 0 and dy < 0:
            move = Move.UP
        else:
            return None
        
        # Check if this move would be a U-turn (which is not allowed)
        if move.value == (-current_direction[0], -current_direction[1]):
            return None
        
        return move
    
    def get_safe_default_move(self, current_pos: Tuple[int, int], 
                            current_direction: Tuple[int, int]) -> Move:
        """Get a safe default move when no path is available."""
        # Try continuing in the current direction first
        forward_pos = (
            current_pos[0] + current_direction[0] * self.cell_size,
            current_pos[1] + current_direction[1] * self.cell_size
        )
        
        # Check if forward move is safe
        if self.is_safe_position(forward_pos):
            return Move(current_direction)
        
        # Get all possible moves (excluding U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        opposite_dir = (-current_direction[0], -current_direction[1])
        valid_moves = [move for move in possible_moves if move.value != opposite_dir]
        
        # Find safe moves
        safe_moves = []
        for move in valid_moves:
            move_dir = move.value
            new_pos = (
                current_pos[0] + move_dir[0] * self.cell_size,
                current_pos[1] + move_dir[1] * self.cell_size
            )
            
            if self.is_safe_position(new_pos):
                safe_moves.append(move)
        
        # If there are safe moves, choose one randomly based on current position
        # to avoid getting stuck in loops
        if safe_moves:
            return safe_moves[hash(str(current_pos)) % len(safe_moves)]
        
        # If no safe moves, just continue in current direction
        return Move(current_direction)
    
    def is_safe_position(self, position: Tuple[int, int]) -> bool:
        """Check if a position is safe (within bounds and not an obstacle)."""
        # Check if position is within game bounds
        if (position[0] < 0 or position[0] >= self.game_width or
            position[1] < 0 or position[1] >= self.game_height):
            return False
        
        # Convert to grid coordinates
        grid_x = position[0] // self.cell_size
        grid_y = position[1] // self.cell_size
        
        # Check if position is an obstacle
        if (0 <= grid_x < self.grid_size[0] and 
            0 <= grid_y < self.grid_size[1] and 
            self.grid[grid_x][grid_y] == 1):
            return False
        
        return True
    
    def get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_in_delivery_zone(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside the delivery zone."""
        x, y = position
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        
        return (x >= dz_pos[0] and x < dz_pos[0] + dz_width and
                y >= dz_pos[1] and y < dz_pos[1] + dz_height)
    
    def check_nearby_passenger(self, current_pos: Tuple[int, int], 
                            current_direction: Tuple[int, int]) -> Optional[Move]:
        """Check if there's a passenger within 3 cells that we can move towards."""
        if not self.passengers:
            return None
        
        # Find the closest passenger within 3 cells
        closest_passenger = None
        closest_distance = float('inf')
        
        for passenger in self.passengers:
            passenger_pos = passenger["position"]
            distance = self.get_distance(current_pos, passenger_pos)
            
            # Check if passenger is within 3 cells
            if distance <= self.cell_size * 3 and distance < closest_distance:
                closest_passenger = passenger
                closest_distance = distance
        
        if not closest_passenger:
            return None
            
        # If passenger is exactly one cell away, pick them up directly
        if closest_distance == self.cell_size:
            passenger_pos = closest_passenger["position"]
            # Calculate the move to get to the passenger
            dx = passenger_pos[0] - current_pos[0]
            dy = passenger_pos[1] - current_pos[1]
            
            # Convert to a Move
            if dx > 0 and dy == 0:
                move = Move.RIGHT
            elif dx < 0 and dy == 0:
                move = Move.LEFT
            elif dx == 0 and dy > 0:
                move = Move.DOWN
            elif dx == 0 and dy < 0:
                move = Move.UP
            else:
                return None
            
            # Check if this move would be a U-turn (which is not allowed)
            if move.value == (-current_direction[0], -current_direction[1]):
                return None
            
            return move
        
        # If passenger is 2-3 cells away, calculate the best move toward them
        passenger_pos = closest_passenger["position"]
        
        # Temporarily clear path to ensure we calculate a fresh path to the passenger
        temp_path = self.path
        self.path = []
        
        # Calculate a quick A* path to the passenger
        pickup_path = self.find_path_astar(current_pos, passenger_pos, current_direction)
        
        # Restore original path
        self.path = temp_path
        
        # If we found a path to the passenger, make the first move
        if pickup_path:
            next_pos = pickup_path[0]
            move = self.get_move_to_adjacent_cell(current_pos, next_pos, current_direction)
            if move:
                # If our current direction is compatible with moving towards the passenger,
                # use it to avoid unnecessary turns
                if self.is_move_towards_target(current_pos, current_direction, passenger_pos):
                    return Move(current_direction)
                return move
        
        return None
        
    def is_move_towards_target(self, current_pos: Tuple[int, int], 
                             current_direction: Tuple[int, int],
                             target_pos: Tuple[int, int]) -> bool:
        """Check if continuing in the current direction gets us closer to the target."""
        # Calculate position after moving in current direction
        next_pos = (
            current_pos[0] + current_direction[0] * self.cell_size,
            current_pos[1] + current_direction[1] * self.cell_size
        )
        
        # Check if this position is valid
        if not self.is_valid_position(next_pos):
            return False
            
        # Calculate current and new distances to target
        current_dist = self.get_distance(current_pos, target_pos)
        next_dist = self.get_distance(next_pos, target_pos)
        
        # If new position is closer to target, this move is towards the target
        return next_dist < current_dist 