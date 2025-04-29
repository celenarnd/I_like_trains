import logging
import math
import heapq
from typing import Tuple, List, Dict, Any, Optional, Set

from common.base_agent import BaseAgent
from common.move import Move

class Agent(BaseAgent):
    """
    Pathfinding Agent
    
    Uses A* algorithm to plan optimal pickup and drop-off paths
    while minimizing collision risks.
    """
    
    def __init__(self, nickname, network, logger="client.agent"):
        super().__init__(nickname, network, logger)
        self.grid_size = None
        self.grid = None
        self.path = []
        self.current_target = None
        self.target_type = None  # "passenger" or "delivery"
    
    def get_move(self) -> Move:
        """Return the best move for the agent based on pathfinding strategy."""
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
        
        # If we're in the delivery zone and have wagons, drop one
        if self.is_in_delivery_zone(current_pos) and train.get("wagons", []):
            return Move.DROP
        
        # Check if we need to recalculate our path
        if (self.needs_new_path(current_pos) or 
            (len(self.path) > 0 and not self.is_safe_path())):
            self.recalculate_path(current_pos, current_direction)
        
        # If we have a path, follow it
        if self.path:
            next_cell = self.path[0]
            next_move = self.get_move_to_adjacent_cell(current_pos, next_cell)
            if next_move:
                # Make sure the move is not opposite to our current direction
                if not self.is_opposite_direction(next_move.value, current_direction):
                    self.path.pop(0)  # Remove the cell we're moving to
                    return next_move
        
        # If we have no path or can't follow it, use a safer approach
        if train.get("wagons", []):
            # We have wagons, target delivery zone
            return self.get_safe_move_towards_delivery(current_pos, current_direction)
        else:
            # We have no wagons, target nearest passenger
            return self.get_safe_move_towards_passenger(current_pos, current_direction)
    
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
    
    def is_safe_path(self) -> bool:
        """Check if the current path is still safe (no new obstacles)."""
        if not self.path:
            return False
            
        # Check if any cell in our path has become an obstacle
        for cell in self.path:
            grid_x = cell[0] // self.cell_size
            grid_y = cell[1] // self.cell_size
            
            # Ensure we're within grid bounds
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1] and 
                self.grid[grid_x][grid_y] == 1):
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
        
        if has_wagons and self.target_type == "passenger":
            # We've picked up a passenger, need to go to delivery zone now
            return True
        elif not has_wagons and self.target_type == "delivery":
            # We've delivered our passengers, need to find a new passenger
            return True
            
        # If our target is a passenger, check if it's still there
        if self.target_type == "passenger" and self.current_target:
            passenger_positions = [p["position"] for p in self.passengers]
            if self.current_target not in passenger_positions:
                return True
                
        return False
    
    def recalculate_path(self, current_pos: Tuple[int, int], current_direction: Tuple[int, int]):
        """Recalculate the path to the appropriate target."""
        train = self.all_trains[self.nickname]
        has_wagons = bool(train.get("wagons", []))
        
        if has_wagons:
            # Head to delivery zone
            self.calculate_path_to_delivery(current_pos, current_direction)
            self.target_type = "delivery"
        else:
            # Head to nearest passenger
            self.calculate_path_to_nearest_passenger(current_pos, current_direction)
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
    
    def calculate_path_to_nearest_passenger(self, current_pos: Tuple[int, int],
                                          current_direction: Tuple[int, int]):
        """Calculate a path to the nearest passenger using A*."""
        if not self.passengers:
            self.path = []
            self.current_target = None
            return
            
        # Find the nearest passenger
        nearest_passenger = min(self.passengers, 
                              key=lambda p: self.get_distance(current_pos, p["position"]))
        
        self.current_target = nearest_passenger["position"]
        self.path = self.find_path_astar(current_pos, self.current_target, current_direction)
    
    def find_path_astar(self, start: Tuple[int, int], goal: Tuple[int, int],
                       current_direction: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a path using A* algorithm."""
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
                    self.grid[neighbor[0]][neighbor[1]] == 1):
                    continue
                
                # Check if the neighbor is in the closed set
                if neighbor in closed_set:
                    continue
                
                # Calculate the tentative g_score
                # Add a turn penalty if we change direction
                turn_penalty = 0
                if direction != current_dir:
                    turn_penalty = 0.5  # Minor penalty for turning
                
                tentative_g = g_score + 1 + turn_penalty
                
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
        
        # Calculate which move gets us closest to the target
        best_move = min(filtered_moves, key=lambda move: 
                       self.get_distance_after_move(current_pos, move.value, dz_center))
        
        return best_move
    
    def get_safe_move_towards_passenger(self, current_pos: Tuple[int, int], 
                                      current_direction: Tuple[int, int]) -> Move:
        """Fallback method for when pathfinding fails."""
        if not self.passengers:
            # No passengers, just keep going in current direction
            return Move(current_direction)
        
        # Find the nearest passenger
        nearest_passenger = min(self.passengers, 
                              key=lambda p: self.get_distance(current_pos, p["position"]))
        
        # Get possible moves (filtering out illegal U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        opposite_dir = (-current_direction[0], -current_direction[1])
        filtered_moves = [move for move in possible_moves 
                        if move.value != opposite_dir]
        
        # Calculate which move gets us closest to the target
        best_move = min(filtered_moves, key=lambda move: 
                       self.get_distance_after_move(current_pos, move.value, 
                                                  nearest_passenger["position"]))
        
        return best_move
    
    def get_distance_after_move(self, current_pos: Tuple[int, int], 
                              move_dir: Tuple[int, int], 
                              target_pos: Tuple[int, int]) -> float:
        """Calculate the distance to the target after making a move."""
        new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                  current_pos[1] + move_dir[1] * self.cell_size)
        return self.get_distance(new_pos, target_pos) 