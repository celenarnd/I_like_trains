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
        
        # Check if an adjacent passenger can be picked up (opportunistic pickup)
        if wagon_count < self.pickup_threshold:
            pickup_move = self.check_adjacent_passenger(current_pos, current_direction)
            if pickup_move:
                self.last_move = pickup_move
                return pickup_move
        
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
        
        # Reset the grid (0 = empty, 1 = obstacle, 2 = danger zone)
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
                
                # Mark cells in front of other trains as dangerous
                for i in range(1, 4):  # Look 3 cells ahead
                    future_x = train_pos[0] + train_dir[0] * self.cell_size * i
                    future_y = train_pos[1] + train_dir[1] * self.cell_size * i
                    
                    grid_x = future_x // self.cell_size
                    grid_y = future_y // self.cell_size
                    
                    # Ensure we're within grid bounds
                    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                        if self.grid[grid_x][grid_y] == 0:  # Don't overwrite obstacles
                            self.grid[grid_x][grid_y] = 2  # Mark as danger zone
    
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
        """Find the best passenger to target based on value and distance."""
        if not self.passengers:
            return None
        
        # Score each passenger based on value and distance
        scored_passengers = []
        for passenger in self.passengers:
            passenger_pos = passenger["position"]
            passenger_value = passenger["value"]
            
            # Check if the passenger's position is an obstacle or danger
            grid_x = passenger_pos[0] // self.cell_size
            grid_y = passenger_pos[1] // self.cell_size
            
            # Skip passengers at obstacle positions
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1] and 
                self.grid[grid_x][grid_y] == 1):
                continue
            
            # Calculate distance to passenger
            distance = self.get_distance(current_pos, passenger_pos)
            
            # Consider nearby passengers as a bonus (cluster pickup)
            nearby_passengers = 0
            for other in self.passengers:
                if other != passenger:
                    other_pos = other["position"]
                    other_dist = self.get_distance(passenger_pos, other_pos)
                    if other_dist <= self.cell_size * 3:  # Within 3 cells
                        nearby_passengers += 1
            
            # Score formula: balance of value, distance, and nearby passengers
            # Higher score is better
            score = (passenger_value + nearby_passengers * 0.5) / max(1, distance / self.cell_size)
            scored_passengers.append((passenger, score))
        
        # If no valid passengers, return None
        if not scored_passengers:
            return None
        
        # Sort by score (higher is better)
        scored_passengers.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest scoring passenger
        return scored_passengers[0][0]
    
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
        """Find a path using A* algorithm with danger awareness."""
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
                
                # Calculate the tentative g_score
                # Add a turn penalty if we change direction
                # Add a danger penalty if the cell is in a danger zone
                turn_penalty = 0
                danger_penalty = 0
                
                if direction != current_dir:
                    turn_penalty = 0.5  # Minor penalty for turning
                
                if (0 <= neighbor[0] < self.grid_size[0] and 
                    0 <= neighbor[1] < self.grid_size[1] and 
                    self.grid[neighbor[0]][neighbor[1]] == 2):  # 2 = danger zone
                    danger_penalty = 2.0  # Significant penalty for danger zones
                
                tentative_g = g_score + 1 + turn_penalty + danger_penalty
                
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