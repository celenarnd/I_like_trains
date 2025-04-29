import logging
import math
from typing import Tuple, List, Dict, Any, Optional, Set

from common.base_agent import BaseAgent
from common.move import Move

class Agent(BaseAgent):
    """
    Defensive Agent
    
    Prioritizes survival by actively avoiding other trains.
    Only picks up passengers when it's safe to do so.
    Focuses on efficient movement and maintaining safe distances.
    Uses diagonal movement by alternating between horizontal and vertical moves.
    """
    
    def __init__(self, nickname, network, logger="client.agent"):
        super().__init__(nickname, network, logger)
        self.last_move = None  # Track last move for diagonal movement
        self.pickup_threshold = 3  # Number of passengers to pick up before delivery
        self.target = None  # Current target (passenger or delivery point)
        self.target_type = None  # "passenger" or "delivery"
        self.safe_distance = 3 * self.cell_size if self.cell_size else 60  # Min distance to other trains
        self.danger_frames = 0  # Counter for consecutive dangerous frames
        self.patience = 20  # How long to wait if unsafe before finding new target
        self.danger_radius = 5  # Cells to mark as dangerous around other trains
    
    def get_move(self) -> Move:
        """Return the next move for the defensive agent."""
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
        
        # Update safe distance based on cell size
        self.safe_distance = 3 * self.cell_size
        
        # Create a danger map of the game area
        danger_map = self.create_danger_map()
        
        # If we're in the delivery zone and have wagons, drop them
        if self.is_in_delivery_zone(current_pos) and wagon_count > 0:
            self.target = None
            self.target_type = None
            return Move.DROP
        
        # If we have enough passengers, head to delivery zone
        if wagon_count >= self.pickup_threshold and self.target_type != "delivery":
            self.target_type = "delivery"
            self.target = self.find_safe_delivery_point(danger_map)
            
        # If we're in immediate danger, make an evasive move
        if self.is_in_danger(current_pos, danger_map):
            self.danger_frames += 1
            evasive_move = self.get_evasive_move(current_pos, current_direction, danger_map)
            if evasive_move:
                self.last_move = evasive_move
                return evasive_move
                
            # If no evasive move is available, just continue in current direction
            self.last_move = Move(current_direction)
            return Move(current_direction)
        else:
            self.danger_frames = 0
            
        # Check if an adjacent passenger can be safely picked up
        if wagon_count < self.pickup_threshold:
            pickup_move = self.check_safe_adjacent_passenger(current_pos, current_direction, danger_map)
            if pickup_move:
                self.last_move = pickup_move
                return pickup_move
                
        # If we need a new target or have been in danger too long, find one
        if ((not self.target) or 
            (self.target_type == "passenger" and self.target not in self.passengers) or
            (self.danger_frames > self.patience)):
            
            # Reset danger counter
            self.danger_frames = 0
            
            if wagon_count >= self.pickup_threshold:
                self.target_type = "delivery"
                self.target = self.find_safe_delivery_point(danger_map)
            else:
                self.target_type = "passenger"
                self.target = self.find_safe_passenger(current_pos, danger_map)
                
        # If we have a target, move towards it
        if self.target:
            target_pos = self.target["position"] if self.target_type == "passenger" else self.target
            
            # Attempt diagonal movement towards target
            diagonal_move = self.move_diagonally(current_pos, target_pos, current_direction, danger_map)
            if diagonal_move:
                return diagonal_move
                
            # If diagonal movement not possible, use direct movement
            return self.direct_move_towards(current_pos, target_pos, current_direction, danger_map)
            
        # If no target, just continue in current direction if safe
        for_move = self.get_safe_forward_move(current_pos, current_direction, danger_map)
        if for_move:
            self.last_move = for_move
            return for_move
            
        # If not safe to go forward, try to find any safe move
        safe_moves = self.get_safe_moves(current_pos, current_direction, danger_map)
        if safe_moves:
            self.last_move = safe_moves[0]
            return safe_moves[0]
            
        # If no safe moves, just continue in current direction
        self.last_move = Move(current_direction)
        return Move(current_direction)
    
    def create_danger_map(self) -> Set[Tuple[int, int]]:
        """Create a set of dangerous grid positions to avoid."""
        danger_points = set()
        
        # Get all train positions and mark them as dangerous
        for train_name, train_data in self.all_trains.items():
            if train_name == self.nickname:
                continue  # Skip our own train
                
            # Mark the train head position as dangerous
            train_pos = train_data["position"]
            grid_x = train_pos[0] // self.cell_size
            grid_y = train_pos[1] // self.cell_size
            danger_points.add((grid_x, grid_y))
            
            # Mark wagon positions as dangerous
            for wagon_pos in train_data.get("wagons", []):
                grid_x = wagon_pos[0] // self.cell_size
                grid_y = wagon_pos[1] // self.cell_size
                danger_points.add((grid_x, grid_y))
                
            # Predict future positions based on direction and mark as dangerous
            train_dir = train_data["direction"]
            for i in range(1, self.danger_radius + 1):
                future_x = train_pos[0] + (train_dir[0] * self.cell_size * i)
                future_y = train_pos[1] + (train_dir[1] * self.cell_size * i)
                
                # Ensure position is within game bounds
                if (0 <= future_x < self.game_width and 0 <= future_y < self.game_height):
                    grid_x = future_x // self.cell_size
                    grid_y = future_y // self.cell_size
                    danger_points.add((grid_x, grid_y))
                    
                    # Mark adjacent cells as dangerous too (with decreasing danger as distance increases)
                    danger_level = max(1, self.danger_radius - i)
                    for dx in range(-danger_level, danger_level + 1):
                        for dy in range(-danger_level, danger_level + 1):
                            if dx == 0 and dy == 0:
                                continue  # Skip the center point (already added)
                            
                            adj_x = grid_x + dx
                            adj_y = grid_y + dy
                            
                            # Ensure position is within game bounds
                            if (0 <= adj_x * self.cell_size < self.game_width and 
                                0 <= adj_y * self.cell_size < self.game_height):
                                danger_points.add((adj_x, adj_y))
        
        return danger_points
    
    def is_in_danger(self, position: Tuple[int, int], danger_map: Set[Tuple[int, int]]) -> bool:
        """Check if the given position is in a dangerous area."""
        grid_x = position[0] // self.cell_size
        grid_y = position[1] // self.cell_size
        
        return (grid_x, grid_y) in danger_map
    
    def get_evasive_move(self, current_pos: Tuple[int, int], current_direction: Tuple[int, int], 
                        danger_map: Set[Tuple[int, int]]) -> Optional[Move]:
        """Get an evasive move to avoid immediate danger."""
        # Get all possible moves (excluding U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        opposite_dir = (-current_direction[0], -current_direction[1])
        valid_moves = [move for move in possible_moves if move.value != opposite_dir]
        
        # Find moves that lead to safety
        safe_moves = []
        for move in valid_moves:
            move_dir = move.value
            new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                      current_pos[1] + move_dir[1] * self.cell_size)
            
            # Ensure we stay within game bounds
            if (0 <= new_pos[0] < self.game_width and 0 <= new_pos[1] < self.game_height):
                grid_x = new_pos[0] // self.cell_size
                grid_y = new_pos[1] // self.cell_size
                
                if (grid_x, grid_y) not in danger_map:
                    safe_moves.append(move)
        
        # If there are safe moves, return one that keeps us farthest from other trains
        if safe_moves:
            safest_move = None
            max_distance = -1
            
            for move in safe_moves:
                move_dir = move.value
                new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                          current_pos[1] + move_dir[1] * self.cell_size)
                
                # Calculate minimum distance to other trains
                min_distance = float('inf')
                for train_name, train_data in self.all_trains.items():
                    if train_name == self.nickname:
                        continue  # Skip our own train
                    
                    train_pos = train_data["position"]
                    distance = self.get_distance(new_pos, train_pos)
                    min_distance = min(min_distance, distance)
                
                if min_distance > max_distance:
                    max_distance = min_distance
                    safest_move = move
            
            return safest_move
        
        return None  # No safe moves available
    
    def check_safe_adjacent_passenger(self, current_pos: Tuple[int, int], 
                                    current_direction: Tuple[int, int], 
                                    danger_map: Set[Tuple[int, int]]) -> Optional[Move]:
        """Check if there's a passenger adjacent to us that we can safely pick up."""
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
                
                # Check if the move is safe
                move_dir = move.value
                new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                          current_pos[1] + move_dir[1] * self.cell_size)
                
                grid_x = new_pos[0] // self.cell_size
                grid_y = new_pos[1] // self.cell_size
                
                if (grid_x, grid_y) not in danger_map:
                    return move
        
        return None
    
    def find_safe_passenger(self, current_pos: Tuple[int, int], 
                           danger_map: Set[Tuple[int, int]]) -> Optional[Dict]:
        """Find a safe passenger to target."""
        if not self.passengers:
            return None
        
        # Calculate safety and value scores for all passengers
        scored_passengers = []
        for passenger in self.passengers:
            passenger_pos = passenger["position"]
            grid_x = passenger_pos[0] // self.cell_size
            grid_y = passenger_pos[1] // self.cell_size
            
            # Check if the passenger is in a dangerous area
            if (grid_x, grid_y) in danger_map:
                continue  # Skip passengers in dangerous areas
            
            distance = self.get_distance(current_pos, passenger_pos)
            value = passenger["value"]
            
            # Calculate safety measure (distance to nearest other train)
            safety = float('inf')
            for train_name, train_data in self.all_trains.items():
                if train_name == self.nickname:
                    continue  # Skip our own train
                
                train_pos = train_data["position"]
                train_distance = self.get_distance(passenger_pos, train_pos)
                safety = min(safety, train_distance)
            
            # If safety is below threshold, skip this passenger
            if safety < self.safe_distance:
                continue
            
            # Score formula: balance of value, distance, and safety
            score = (value * safety) / (distance + 1)
            scored_passengers.append((passenger, score))
        
        # If no safe passengers, return None
        if not scored_passengers:
            return None
        
        # Sort by score (higher is better)
        scored_passengers.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest scoring passenger
        return scored_passengers[0][0]
    
    def find_safe_delivery_point(self, danger_map: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Find a safe point in the delivery zone."""
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        
        # Check all points around the delivery zone edge for safety
        safe_points = []
        
        # Check horizontal edges
        for x in range(dz_pos[0], dz_pos[0] + dz_width, self.cell_size):
            # Top edge
            y = dz_pos[1]
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            if (grid_x, grid_y) not in danger_map:
                safe_points.append((x, y))
            
            # Bottom edge
            y = dz_pos[1] + dz_height - self.cell_size
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            if (grid_x, grid_y) not in danger_map:
                safe_points.append((x, y))
        
        # Check vertical edges
        for y in range(dz_pos[1], dz_pos[1] + dz_height, self.cell_size):
            # Left edge
            x = dz_pos[0]
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            if (grid_x, grid_y) not in danger_map:
                safe_points.append((x, y))
            
            # Right edge
            x = dz_pos[0] + dz_width - self.cell_size
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            if (grid_x, grid_y) not in danger_map:
                safe_points.append((x, y))
        
        # If no safe points found, use center of delivery zone
        if not safe_points:
            return (dz_pos[0] + dz_width // 2, dz_pos[1] + dz_height // 2)
        
        # Calculate the safest point (farthest from other trains)
        safest_point = None
        max_safety = -1
        
        for point in safe_points:
            # Calculate minimum distance to other trains
            min_distance = float('inf')
            for train_name, train_data in self.all_trains.items():
                if train_name == self.nickname:
                    continue  # Skip our own train
                
                train_pos = train_data["position"]
                distance = self.get_distance(point, train_pos)
                min_distance = min(min_distance, distance)
            
            if min_distance > max_safety:
                max_safety = min_distance
                safest_point = point
        
        return safest_point
    
    def move_diagonally(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                       current_direction: Tuple[int, int], 
                       danger_map: Set[Tuple[int, int]]) -> Optional[Move]:
        """Implement diagonal movement by alternating horizontal and vertical moves."""
        # Calculate the direction to the target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # If we're already at the target, no need to move
        if dx == 0 and dy == 0:
            return None
        
        # Determine which directions are valid (not opposite to current direction)
        valid_moves = []
        opposite_dir = (-current_direction[0], -current_direction[1])
        
        if dx > 0 and opposite_dir != (1, 0):
            valid_moves.append(Move.RIGHT)
        if dx < 0 and opposite_dir != (-1, 0):
            valid_moves.append(Move.LEFT)
        if dy > 0 and opposite_dir != (0, 1):
            valid_moves.append(Move.DOWN)
        if dy < 0 and opposite_dir != (0, -1):
            valid_moves.append(Move.UP)
        
        # Filter out unsafe moves
        safe_moves = []
        for move in valid_moves:
            move_dir = move.value
            new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                      current_pos[1] + move_dir[1] * self.cell_size)
            
            # Ensure we stay within game bounds
            if (0 <= new_pos[0] < self.game_width and 0 <= new_pos[1] < self.game_height):
                grid_x = new_pos[0] // self.cell_size
                grid_y = new_pos[1] // self.cell_size
                
                if (grid_x, grid_y) not in danger_map:
                    safe_moves.append(move)
        
        if not safe_moves:
            return None
        
        # For diagonal movement, alternate between horizontal and vertical
        if self.last_move:
            # If last move was horizontal, try vertical next (if possible)
            if self.last_move in [Move.LEFT, Move.RIGHT]:
                vertical_moves = [move for move in safe_moves if move in [Move.UP, Move.DOWN]]
                if vertical_moves:
                    self.last_move = vertical_moves[0]
                    return vertical_moves[0]
            # If last move was vertical, try horizontal next (if possible)
            else:
                horizontal_moves = [move for move in safe_moves if move in [Move.LEFT, Move.RIGHT]]
                if horizontal_moves:
                    self.last_move = horizontal_moves[0]
                    return horizontal_moves[0]
        
        # If no alternating move is possible or no last move, just pick the best direction
        best_move = self.get_best_move(current_pos, target_pos, safe_moves)
        self.last_move = best_move
        return best_move
    
    def direct_move_towards(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                          current_direction: Tuple[int, int], 
                          danger_map: Set[Tuple[int, int]]) -> Move:
        """Make a direct move towards the target position, avoiding danger."""
        # Get all possible moves (excluding U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        opposite_dir = (-current_direction[0], -current_direction[1])
        valid_moves = [move for move in possible_moves if move.value != opposite_dir]
        
        # Filter out unsafe moves
        safe_moves = []
        for move in valid_moves:
            move_dir = move.value
            new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                      current_pos[1] + move_dir[1] * self.cell_size)
            
            # Ensure we stay within game bounds
            if (0 <= new_pos[0] < self.game_width and 0 <= new_pos[1] < self.game_height):
                grid_x = new_pos[0] // self.cell_size
                grid_y = new_pos[1] // self.cell_size
                
                if (grid_x, grid_y) not in danger_map:
                    safe_moves.append(move)
        
        # If there are safe moves, choose the best one
        if safe_moves:
            best_move = self.get_best_move(current_pos, target_pos, safe_moves)
            self.last_move = best_move
            return best_move
        
        # If no safe moves, try to find the least dangerous one
        if valid_moves:
            safest_move = None
            max_distance = -1
            
            for move in valid_moves:
                move_dir = move.value
                new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                          current_pos[1] + move_dir[1] * self.cell_size)
                
                # Skip moves that would go out of bounds
                if (new_pos[0] < 0 or new_pos[0] >= self.game_width or
                    new_pos[1] < 0 or new_pos[1] >= self.game_height):
                    continue
                
                # Calculate minimum distance to other trains
                min_distance = float('inf')
                for train_name, train_data in self.all_trains.items():
                    if train_name == self.nickname:
                        continue  # Skip our own train
                    
                    train_pos = train_data["position"]
                    distance = self.get_distance(new_pos, train_pos)
                    min_distance = min(min_distance, distance)
                
                if min_distance > max_distance:
                    max_distance = min_distance
                    safest_move = move
            
            if safest_move:
                self.last_move = safest_move
                return safest_move
        
        # If still no move found, just continue in current direction
        self.last_move = Move(current_direction)
        return Move(current_direction)
    
    def get_best_move(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                     valid_moves: List[Move]) -> Optional[Move]:
        """Get the best move from a list of valid moves to get closer to the target."""
        best_move = None
        min_distance = float('inf')
        
        for move in valid_moves:
            # Calculate new position after move
            move_dir = move.value
            new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                      current_pos[1] + move_dir[1] * self.cell_size)
            
            # Calculate new distance to target
            new_distance = self.get_distance(new_pos, target_pos)
            
            # Check if this is the best move so far
            if new_distance < min_distance:
                min_distance = new_distance
                best_move = move
        
        # If no best move found, just return the first valid move
        return best_move if best_move else valid_moves[0] if valid_moves else None
    
    def get_safe_moves(self, current_pos: Tuple[int, int], current_direction: Tuple[int, int], 
                      danger_map: Set[Tuple[int, int]]) -> List[Move]:
        """Get all safe moves from the current position."""
        # Get all possible moves (excluding U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        opposite_dir = (-current_direction[0], -current_direction[1])
        valid_moves = [move for move in possible_moves if move.value != opposite_dir]
        
        # Filter out unsafe moves
        safe_moves = []
        for move in valid_moves:
            move_dir = move.value
            new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                      current_pos[1] + move_dir[1] * self.cell_size)
            
            # Ensure we stay within game bounds
            if (0 <= new_pos[0] < self.game_width and 0 <= new_pos[1] < self.game_height):
                grid_x = new_pos[0] // self.cell_size
                grid_y = new_pos[1] // self.cell_size
                
                if (grid_x, grid_y) not in danger_map:
                    safe_moves.append(move)
        
        return safe_moves
    
    def get_safe_forward_move(self, current_pos: Tuple[int, int], current_direction: Tuple[int, int], 
                            danger_map: Set[Tuple[int, int]]) -> Optional[Move]:
        """Check if it's safe to continue in the current direction."""
        forward_move = Move(current_direction)
        new_pos = (current_pos[0] + current_direction[0] * self.cell_size, 
                  current_pos[1] + current_direction[1] * self.cell_size)
        
        # Ensure we stay within game bounds
        if (0 <= new_pos[0] < self.game_width and 0 <= new_pos[1] < self.game_height):
            grid_x = new_pos[0] // self.cell_size
            grid_y = new_pos[1] // self.cell_size
            
            if (grid_x, grid_y) not in danger_map:
                return forward_move
        
        return None
    
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