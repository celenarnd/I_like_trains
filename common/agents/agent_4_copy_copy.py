import logging
import math
import heapq
from typing import Tuple, List, Dict, Any, Optional, Set

from common.base_agent import BaseAgent
from common.move import Move

class Agent(BaseAgent):
    """
    Optimal Multi-Passenger Agent
    
    Strategically handles multiple passengers by evaluating optimal routes
    and planning paths that maximize efficiency while minimizing risk.
    Takes at least 5 passengers before delivering and uses diagonal movement.
    """
    
    def __init__(self, nickname, network, logger="client.agent"):
        super().__init__(nickname, network, logger)
        self.grid_size = None
        self.grid = None
        self.path = []
        self.current_target = None
        self.target_type = None  # "passenger" or "delivery"
        self.passenger_value_threshold = 2  # Minimum passenger value worth pursuing
        self.path_recalculation_frequency = 10  # Recalculate path every N frames
        self.frame_counter = 0
        self.nearby_pickup_radius = 3  # Cell radius to consider picking up multiple passengers
        self.min_passengers_before_delivery = 4  # Collect at least this many passengers before delivery
        self.last_move = None  # Track last move for diagonal movement
        self.diagonal_movement = True  # Enable diagonal movement
        self.stay_at_delivery_until_empty = True  # Stay at delivery zone until all passengers are dropped
    
    def get_move(self) -> Move:
        """Return the best move for the agent based on optimal multi-passenger strategy."""
        # Make sure we have all the needed game state
        if not all([self.all_trains, self.passengers, self.delivery_zone, 
                   self.game_width, self.game_height, self.cell_size]):
            return Move.RIGHT  # Default move if we don't have all the info
            
        # Get our train details
        if self.nickname not in self.all_trains:
            return Move.RIGHT
            
        train = self.all_trains[self.nickname]
        current_pos = train["position"]
        current_direction = train["direction"]
        
        # Update our grid representation of the game
        self.update_grid()
        
        # Increment frame counter for periodic path recalculation
        self.frame_counter = (self.frame_counter + 1) % self.path_recalculation_frequency
        
        # If we're in the delivery zone and have wagons, drop them all off one by one
        if self.is_in_delivery_zone(current_pos) and train.get("wagons", []):
            return Move.DROP
            
        # Check if we need to recalculate our path
        if (self.needs_new_path(current_pos) or 
            self.frame_counter == 0 or  # Periodic recalculation
            (len(self.path) > 0 and not self.is_safe_path())):
            self.recalculate_path(current_pos, current_direction)
        
        # Check if there's a passenger adjacent to us - opportunistic pickup
        adjacent_passenger = self.check_adjacent_passenger(current_pos)
        if adjacent_passenger:
            wagon_count = len(train.get("wagons", []))
            # Get the move to collect this passenger
            passenger_pos = adjacent_passenger["position"]
            move_to_passenger = self.get_move_to_adjacent_cell(current_pos, passenger_pos)
            if move_to_passenger and not self.is_opposite_direction(move_to_passenger.value, current_direction):
                return move_to_passenger
        
        # If we have a path, follow it with diagonal movement if possible
        if self.path:
            # If diagonal movement is enabled, try to move diagonally
            if self.diagonal_movement:
                next_move = self.get_diagonal_move(current_pos, current_direction)
                if next_move:
                    return next_move
            
            # If diagonal movement not possible or not enabled, use normal movement
            next_cell = self.path[0]
            next_move = self.get_move_to_adjacent_cell(current_pos, next_cell)
            if next_move:
                # Make sure the move is not opposite to our current direction
                if not self.is_opposite_direction(next_move.value, current_direction):
                    self.path.pop(0)  # Remove the cell we're moving to
                    self.last_move = next_move
                    return next_move
        
        # If we have no path or can't follow it, use a safer approach
        wagon_count = len(train.get("wagons", []))
        if wagon_count >= self.min_passengers_before_delivery:
            # We have enough wagons, target delivery zone
            return self.get_safe_move_towards_delivery(current_pos, current_direction)
        else:
            # We need more passengers, target the optimal passenger
            return self.get_safe_move_towards_best_passenger(current_pos, current_direction)
    
    def get_diagonal_move(self, current_pos: Tuple[int, int], current_direction: Tuple[int, int]) -> Optional[Move]:
        """Attempt to make a diagonal move by alternating between two directions."""
        if len(self.path) < 2:
            return None
            
        # Get the next two cells in the path
        next_cell = self.path[0]
        next_next_cell = self.path[1]
        
        # Calculate the direction vectors
        dx1 = next_cell[0] - current_pos[0]
        dy1 = next_cell[1] - current_pos[1]
        dx2 = next_next_cell[0] - next_cell[0]
        dy2 = next_next_cell[1] - next_cell[1]
        
        # Normalize the directions
        if dx1 != 0:
            dx1 = dx1 // abs(dx1) * self.cell_size
        if dy1 != 0:
            dy1 = dy1 // abs(dy1) * self.cell_size
        if dx2 != 0:
            dx2 = dx2 // abs(dx2) * self.cell_size
        if dy2 != 0:
            dy2 = dy2 // abs(dy2) * self.cell_size
        
        # Check if we're moving in different directions over the next two cells
        if (dx1 != 0 and dy2 != 0) or (dy1 != 0 and dx2 != 0):
            # We have a potential diagonal move
            
            # If last move was horizontal, try vertical and vice versa
            if self.last_move:
                if self.last_move == Move.RIGHT or self.last_move == Move.LEFT:
                    # Last move was horizontal, try vertical
                    if dy1 != 0:
                        move = Move.DOWN if dy1 > 0 else Move.UP
                        self.last_move = move
                        return move
                    elif dy2 != 0:
                        move = Move.DOWN if dy2 > 0 else Move.UP
                        self.last_move = move
                        return move
                else:
                    # Last move was vertical, try horizontal
                    if dx1 != 0:
                        move = Move.RIGHT if dx1 > 0 else Move.LEFT
                        self.last_move = move
                        return move
                    elif dx2 != 0:
                        move = Move.RIGHT if dx2 > 0 else Move.LEFT
                        self.last_move = move
                        return move
            else:
                # No last move, start with either direction
                if dx1 != 0:
                    move = Move.RIGHT if dx1 > 0 else Move.LEFT
                    self.last_move = move
                    return move
                elif dy1 != 0:
                    move = Move.DOWN if dy1 > 0 else Move.UP
                    self.last_move = move
                    return move
        
        return None
    
    def update_grid(self):
        """Update the grid representation of the game."""
        # Calculate grid size based on cell_size
        grid_width = self.game_width // self.cell_size
        grid_height = self.game_height // self.cell_size
        
        if self.grid_size != (grid_width, grid_height):
            self.grid_size = (grid_width, grid_height)
            self.grid = [[0 for _ in range(grid_height)] for _ in range(grid_width)]
        
        # Reset the grid
        for x in range(grid_width):
            for y in range(grid_height):
                self.grid[x][y] = 0
        
        # Mark obstacles (trains and wagons)
        for train_name, other_train in self.all_trains.items():
            # Convert pixel position to grid position
            train_pos = other_train["position"]
            grid_x = train_pos[0] // self.cell_size
            grid_y = train_pos[1] // self.cell_size
            
            # Ensure we're within grid bounds
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                # Mark position as obstacle
                self.grid[grid_x][grid_y] = 1
            
            # Mark wagon positions as obstacles
            for wagon_pos in other_train.get("wagons", []):
                grid_x = wagon_pos[0] // self.cell_size
                grid_y = wagon_pos[1] // self.cell_size
                
                # Ensure we're within grid bounds
                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    self.grid[grid_x][grid_y] = 1
        
        # Also mark the areas near other train heads as higher risk
        for train_name, other_train in self.all_trains.items():
            # Skip our own train
            if train_name == self.nickname:
                continue
                
            # Get the train's position and direction
            train_pos = other_train["position"]
            train_dir = other_train["direction"]
            
            # Mark cells in front of other trains as dangerous (extrapolate movement)
            for i in range(1, 4):  # Look 3 cells ahead
                danger_pos = (
                    train_pos[0] + train_dir[0] * self.cell_size * i,
                    train_pos[1] + train_dir[1] * self.cell_size * i
                )
                
                grid_x = danger_pos[0] // self.cell_size
                grid_y = danger_pos[1] // self.cell_size
                
                # Ensure we're within grid bounds
                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    # Set as danger zone (we'll treat this as higher cost in pathfinding)
                    if self.grid[grid_x][grid_y] == 0:  # Don't overwrite obstacles
                        self.grid[grid_x][grid_y] = 2  # 2 = danger zone
    
    def check_adjacent_passenger(self, current_pos: Tuple[int, int]) -> Optional[Dict]:
        """Check if there's a passenger adjacent to current position."""
        if not self.passengers:
            return None
            
        # Check each passenger
        for passenger in self.passengers:
            passenger_pos = passenger["position"]
            distance = self.get_distance(current_pos, passenger_pos)
            
            # If passenger is exactly one cell away (Manhattan distance = cell_size)
            if distance == self.cell_size:
                return passenger
                
        return None
    
    def is_safe_path(self) -> bool:
        """Check if the current path is still safe (no new obstacles or dangers)."""
        if not self.path:
            return False
            
        # Check if any cell in our path has become an obstacle or danger
        for cell in self.path:
            grid_x = cell[0] // self.cell_size
            grid_y = cell[1] // self.cell_size
            
            # Ensure we're within grid bounds
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1] and 
                self.grid[grid_x][grid_y] > 0):  # Either obstacle or danger zone
                return False
        
        return True
    
    def needs_new_path(self, current_pos: Tuple[int, int]) -> bool:
        """Decide if we need to recalculate the path."""
        # If we have no path, we need a new one
        if not self.path:
            return True
            
        # If we have a path but our target has changed, we need a new one
        train = self.all_trains[self.nickname]
        has_wagons = bool(train.get("wagons", []))
        wagon_count = len(train.get("wagons", []))
        
        # Check if we've just collected enough passengers to go to delivery
        if has_wagons and wagon_count >= self.min_passengers_before_delivery and self.target_type == "passenger":
            return True
            
        # Check if we've just delivered passengers and need to collect more
        if not has_wagons and self.target_type == "delivery":
            return True
            
        # If our target is a passenger, check if it's still there
        if self.target_type == "passenger" and self.current_target:
            passenger_positions = [p["position"] for p in self.passengers]
            if self.current_target not in passenger_positions:
                return True
        
        # Check if we can find a more valuable passenger nearby
        if has_wagons and wagon_count < self.min_passengers_before_delivery and self.target_type == "passenger":
            current_target_value = 0
            for p in self.passengers:
                if p["position"] == self.current_target:
                    current_target_value = p["value"]
                    break
                    
            better_passenger = self.find_better_passenger(current_pos, current_target_value)
            if better_passenger:
                return True
                
        return False
    
    def find_better_passenger(self, current_pos: Tuple[int, int], current_value: int) -> bool:
        """Check if there's a more valuable passenger within a reasonable distance."""
        if not self.passengers:
            return False
            
        for passenger in self.passengers:
            # If this passenger is more valuable than our current target
            if passenger["value"] > current_value + 1:  # Significantly more valuable
                # And it's within reasonable distance
                distance = self.get_distance(current_pos, passenger["position"])
                if distance < self.cell_size * 5:  # Within 5 cells
                    return True
                    
        return False
    
    def recalculate_path(self, current_pos: Tuple[int, int], current_direction: Tuple[int, int]):
        """Recalculate the path to the appropriate target."""
        train = self.all_trains[self.nickname]
        has_wagons = bool(train.get("wagons", []))
        wagon_count = len(train.get("wagons", []))
        
        if has_wagons and wagon_count >= self.min_passengers_before_delivery:
            # We have enough wagons, head to delivery zone
            self.calculate_path_to_delivery(current_pos, current_direction)
            self.target_type = "delivery"
        else:
            # We need more passengers, head to optimal passenger
            self.calculate_path_to_optimal_passenger(current_pos, current_direction)
            self.target_type = "passenger"
    
    def calculate_path_to_delivery(self, current_pos: Tuple[int, int], 
                                 current_direction: Tuple[int, int]):
        """Calculate a path to the delivery zone using A*."""
        # Calculate the center of the delivery zone
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        
        # Find the nearest point in the delivery zone
        best_distance = float('inf')
        best_target = None
        
        # Check all edge cells of the delivery zone
        for x in range(dz_pos[0], dz_pos[0] + dz_width, self.cell_size):
            for y in range(dz_pos[1], dz_pos[1] + dz_height, self.cell_size):
                # Only consider the edge cells
                if (x == dz_pos[0] or x == dz_pos[0] + dz_width - self.cell_size or
                    y == dz_pos[1] or y == dz_pos[1] + dz_height - self.cell_size):
                    dist = self.get_distance(current_pos, (x, y))
                    if dist < best_distance:
                        best_distance = dist
                        best_target = (x, y)
        
        if best_target:
            self.current_target = best_target
            self.path = self.find_path_astar(current_pos, best_target, current_direction)
    
    def calculate_path_to_optimal_passenger(self, current_pos: Tuple[int, int],
                                          current_direction: Tuple[int, int]):
        """Calculate a path to the optimal passenger using A*."""
        if not self.passengers:
            self.path = []
            self.current_target = None
            return
        
        # Create a list of (passenger, score) tuples
        scored_passengers = []
        for passenger in self.passengers:
            passenger_pos = passenger["position"]
            passenger_value = passenger["value"]
            
            # Calculate base distance to passenger
            distance = self.get_distance(current_pos, passenger_pos)
            
            # Check how many other passengers are nearby this one
            nearby_passengers = 0
            for other_passenger in self.passengers:
                if other_passenger["position"] != passenger_pos:
                    other_pos = other_passenger["position"]
                    if self.get_distance(passenger_pos, other_pos) <= self.cell_size * self.nearby_pickup_radius:
                        nearby_passengers += 1
            
            # Calculate a score based on value, distance, and nearby passengers
            # Higher value, shorter distance, and more nearby passengers = better score
            score = (passenger_value + nearby_passengers * 0.5) / (distance + 1)
            
            scored_passengers.append((passenger, score))
        
        # Sort by score (highest first)
        scored_passengers.sort(key=lambda x: x[1], reverse=True)
        
        # Take the best passenger
        best_passenger = scored_passengers[0][0]
        self.current_target = best_passenger["position"]
        
        # Find a path to this passenger
        self.path = self.find_path_astar(current_pos, self.current_target, current_direction)
    
    def find_path_astar(self, start: Tuple[int, int], goal: Tuple[int, int],
                       current_direction: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a path using A* algorithm with improved heuristics."""
        # Convert positions to grid coordinates
        start_grid = (start[0] // self.cell_size, start[1] // self.cell_size)
        goal_grid = (goal[0] // self.cell_size, goal[1] // self.cell_size)
        
        # Check if goal is an obstacle
        if (0 <= goal_grid[0] < self.grid_size[0] and 
            0 <= goal_grid[1] < self.grid_size[1] and 
            self.grid[goal_grid[0]][goal_grid[1]] == 1):
            # Goal is blocked, find a nearby accessible cell
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    
                    alt_goal_x = goal_grid[0] + dx
                    alt_goal_y = goal_grid[1] + dy
                    
                    if (0 <= alt_goal_x < self.grid_size[0] and 
                        0 <= alt_goal_y < self.grid_size[1] and 
                        self.grid[alt_goal_x][alt_goal_y] == 0):
                        goal_grid = (alt_goal_x, alt_goal_y)
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
                path = self.reconstruct_path(parents, current, start_grid)
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
    
    def reconstruct_path(self, parents, current, start):
        """Reconstruct the path from the goal to the start."""
        path = [current]
        while current in parents and current != start:
            current = parents[current]
            path.append(current)
        path.reverse()
        # Remove the first element (current position)
        if path and path[0] == start:
            path.pop(0)
        return path
    
    def get_move_to_adjacent_cell(self, current_pos: Tuple[int, int], 
                                 next_pos: Tuple[int, int]) -> Optional[Move]:
        """Get the move to go from current position to an adjacent cell."""
        # Calculate the direction vector
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        # Normalize the direction to match cell size
        if dx != 0:
            dx = dx // abs(dx) * self.cell_size
        if dy != 0:
            dy = dy // abs(dy) * self.cell_size
        
        # Convert to a Move
        if dx > 0 and dy == 0:
            return Move.RIGHT
        elif dx < 0 and dy == 0:
            return Move.LEFT
        elif dx == 0 and dy > 0:
            return Move.DOWN
        elif dx == 0 and dy < 0:
            return Move.UP
        
        return None
    
    def is_opposite_direction(self, new_direction: Tuple[int, int], 
                            current_direction: Tuple[int, int]) -> bool:
        """Check if the new direction is opposite to the current direction."""
        return (new_direction[0] == -current_direction[0] and 
                new_direction[1] == -current_direction[1])
    
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
    
    def get_safe_move_towards_delivery(self, current_pos: Tuple[int, int], 
                                     current_direction: Tuple[int, int]) -> Move:
        """Fallback method for when pathfinding fails."""
        # Calculate the center of the delivery zone
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        dz_center = (dz_pos[0] + dz_width // 2, dz_pos[1] + dz_height // 2)
        
        # Get possible moves (filtering out illegal U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        opposite_dir = (-current_direction[0], -current_direction[1])
        filtered_moves = [move for move in possible_moves 
                        if move.value != opposite_dir]
        
        # Filter out dangerous moves that lead to immediate collisions
        safe_moves = []
        for move in filtered_moves:
            if not self.check_immediate_danger(current_pos, move.value):
                safe_moves.append(move)
        
        if not safe_moves:
            safe_moves = filtered_moves  # If no safe moves, use all available moves
        
        # Calculate which move gets us closest to the target
        best_move = min(safe_moves, key=lambda move: 
                       self.get_distance_after_move(current_pos, move.value, dz_center))
        
        self.last_move = best_move  # Update last move for diagonal movement
        return best_move
    
    def get_safe_move_towards_best_passenger(self, current_pos: Tuple[int, int], 
                                           current_direction: Tuple[int, int]) -> Move:
        """Fallback method for when pathfinding fails."""
        if not self.passengers:
            # No passengers, just keep going in current direction but avoid collisions
            possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
            opposite_dir = (-current_direction[0], -current_direction[1])
            filtered_moves = [move for move in possible_moves 
                            if move.value != opposite_dir]
            
            # Filter out dangerous moves
            safe_moves = []
            for move in filtered_moves:
                if not self.check_immediate_danger(current_pos, move.value):
                    safe_moves.append(move)
            
            if safe_moves:
                self.last_move = safe_moves[0]  # Update last move for diagonal movement
                return safe_moves[0]  # Just take the first safe move
            else:
                self.last_move = Move(current_direction)  # Update last move for diagonal movement
                return Move(current_direction)  # If no safe moves, continue in current direction
        
        # Calculate scores for each passenger based on value and distance
        scored_passengers = []
        for passenger in self.passengers:
            passenger_pos = passenger["position"]
            passenger_value = passenger["value"]
            
            # Calculate base distance to passenger
            distance = self.get_distance(current_pos, passenger_pos)
            
            # Check how many other passengers are nearby this one
            nearby_passengers = 0
            for other_passenger in self.passengers:
                if other_passenger["position"] != passenger_pos:
                    other_pos = other_passenger["position"]
                    if self.get_distance(passenger_pos, other_pos) <= self.cell_size * self.nearby_pickup_radius:
                        nearby_passengers += 1
            
            # Calculate a score based on value, distance, and nearby passengers
            score = (passenger_value + nearby_passengers * 0.5) / (distance + 1)
            
            scored_passengers.append((passenger, score))
        
        # Sort by score (highest first)
        scored_passengers.sort(key=lambda x: x[1], reverse=True)
        
        # Take the best passenger
        best_passenger = scored_passengers[0][0]
        best_passenger_pos = best_passenger["position"]
        
        # Get possible moves (filtering out illegal U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        opposite_dir = (-current_direction[0], -current_direction[1])
        filtered_moves = [move for move in possible_moves 
                        if move.value != opposite_dir]
        
        # Filter out dangerous moves
        safe_moves = []
        for move in filtered_moves:
            if not self.check_immediate_danger(current_pos, move.value):
                safe_moves.append(move)
        
        if not safe_moves:
            safe_moves = filtered_moves  # If no safe moves, use all available moves
        
        # Calculate which move gets us closest to the target
        best_move = min(safe_moves, key=lambda move: 
                       self.get_distance_after_move(current_pos, move.value, best_passenger_pos))
        
        self.last_move = best_move  # Update last move for diagonal movement
        return best_move
    
    def check_immediate_danger(self, position: Tuple[int, int], 
                              direction: Tuple[int, int]) -> bool:
        """Check if moving in this direction would lead to immediate collision."""
        # Calculate the next position
        next_pos = (
            position[0] + direction[0] * self.cell_size,
            position[1] + direction[1] * self.cell_size
        )
        
        # Check for out of bounds
        if (next_pos[0] < 0 or next_pos[0] >= self.game_width or
            next_pos[1] < 0 or next_pos[1] >= self.game_height):
            return True
        
        # Check for collision with trains or wagons
        for train_name, other_train in self.all_trains.items():
            # Skip our own train
            if train_name == self.nickname:
                continue
                
            # Check if another train's head is at the next position
            if other_train["position"] == next_pos:
                return True
                
            # Check if any wagon is at the next position
            for wagon_pos in other_train.get("wagons", []):
                if wagon_pos == next_pos:
                    return True
        
        return False
    
    def get_distance_after_move(self, current_pos: Tuple[int, int], 
                              move_dir: Tuple[int, int], 
                              target_pos: Tuple[int, int]) -> float:
        """Calculate the distance to the target after making a move."""
        new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                  current_pos[1] + move_dir[1] * self.cell_size)
        return self.get_distance(new_pos, target_pos) 