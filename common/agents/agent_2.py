import logging
import math
from typing import Tuple, List, Dict, Any, Optional

from common.base_agent import BaseAgent
from common.move import Move

class Agent(BaseAgent):
    """
    Defensive Agent
    
    Prioritizes survival by avoiding other trains and only picks up passengers when safe.
    Focuses on efficient movement and collision avoidance.
    """
    
    def get_move(self) -> Move:
        """Return the best move for the agent based on the defensive strategy."""
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
        
        # If we're in the delivery zone and have wagons, drop one
        if self.is_in_delivery_zone(current_pos) and train.get("wagons", []):
            return Move.DROP
        
        # Check for potential collisions in our current direction
        danger_ahead = self.check_danger_ahead(current_pos, current_direction)
        
        # If there's danger ahead, find a safe direction
        if danger_ahead:
            safe_direction = self.find_safe_direction(current_pos, current_direction)
            if safe_direction:
                return safe_direction
        
        # If we have wagons, head to delivery zone (but safely)
        if train.get("wagons", []):
            return self.get_safe_move_towards_delivery()
        
        # No wagons, go to nearest passenger if it's safe to do so
        return self.get_safe_move_towards_passenger()
    
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
    
    def check_danger_ahead(self, position: Tuple[int, int], direction: Tuple[int, int], 
                          look_ahead: int = 3) -> bool:
        """Check if there's a train or wagon in our path within n cells."""
        # Calculate positions ahead
        positions_ahead = []
        for i in range(1, look_ahead + 1):
            pos_ahead = (
                position[0] + direction[0] * self.cell_size * i,
                position[1] + direction[1] * self.cell_size * i
            )
            positions_ahead.append(pos_ahead)
        
        # Check for wall collision
        for pos in positions_ahead:
            if (pos[0] < 0 or pos[0] >= self.game_width or 
                pos[1] < 0 or pos[1] >= self.game_height):
                return True
        
        # Check for train/wagon collision
        for train_name, other_train in self.all_trains.items():
            # Skip our own train
            if train_name == self.nickname:
                continue
                
            # Check if another train's head is in our path
            if other_train["position"] in positions_ahead:
                return True
                
            # Check if any wagon is in our path
            for wagon_pos in other_train.get("wagons", []):
                if wagon_pos in positions_ahead:
                    return True
        
        # No danger detected
        return False
    
    def find_safe_direction(self, position: Tuple[int, int], current_direction: Tuple[int, int]) -> Optional[Move]:
        """Find a safe direction to turn to avoid collision."""
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        
        # Filter out the opposite of our current direction (can't make U-turns)
        opposite_dir = (-current_direction[0], -current_direction[1])
        filtered_moves = [move for move in possible_moves 
                         if move.value != opposite_dir]
        
        # Check each direction for safety
        safe_moves = []
        for move in filtered_moves:
            if not self.check_danger_ahead(position, move.value):
                safe_moves.append(move)
        
        if not safe_moves:
            # No safe moves available, try to pick the "least dangerous" one
            # by looking one cell ahead only
            for move in filtered_moves:
                next_pos = (
                    position[0] + move.value[0] * self.cell_size,
                    position[1] + move.value[1] * self.cell_size
                )
                
                # Check if this position is within bounds
                if (next_pos[0] >= 0 and next_pos[0] < self.game_width and 
                    next_pos[1] >= 0 and next_pos[1] < self.game_height):
                    # Check if this position collides with any train or wagon
                    collision = False
                    for train_name, other_train in self.all_trains.items():
                        if train_name == self.nickname:
                            continue
                        if other_train["position"] == next_pos:
                            collision = True
                            break
                        for wagon_pos in other_train.get("wagons", []):
                            if wagon_pos == next_pos:
                                collision = True
                                break
                        if collision:
                            break
                    
                    if not collision:
                        return move
            
            # Still no safe move found, just continue in current direction
            return Move(current_direction)
        
        # If we have a delivery zone target or passenger target, pick the safe move
        # that gets us closest to our objective
        train = self.all_trains[self.nickname]
        if train.get("wagons", []):
            # We have wagons, target delivery zone
            dz_pos = self.delivery_zone["position"]
            dz_width = self.delivery_zone["width"]
            dz_height = self.delivery_zone["height"]
            dz_center = (dz_pos[0] + dz_width // 2, dz_pos[1] + dz_height // 2)
            
            return min(safe_moves, key=lambda move: 
                      self.get_distance_after_move(position, move.value, dz_center))
        elif self.passengers:
            # Target nearest passenger
            nearest_passenger = min(self.passengers, 
                                  key=lambda p: self.get_distance(position, p["position"]))
            passenger_pos = nearest_passenger["position"]
            
            return min(safe_moves, key=lambda move: 
                      self.get_distance_after_move(position, move.value, passenger_pos))
        
        # No specific target, just pick any safe move
        return safe_moves[0]
    
    def get_safe_move_towards_delivery(self) -> Move:
        """Get a safe move towards the delivery zone."""
        train = self.all_trains[self.nickname]
        current_pos = train["position"]
        current_direction = train["direction"]
        
        # Calculate the center of the delivery zone
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        dz_center = (dz_pos[0] + dz_width // 2, dz_pos[1] + dz_height // 2)
        
        # Get best move towards delivery zone
        best_move = self.get_next_move_towards(current_pos, current_direction, dz_center)
        
        # Check if the best move is safe
        if not self.check_danger_ahead(current_pos, best_move.value):
            return best_move
        
        # If not safe, find an alternative safe direction
        safe_direction = self.find_safe_direction(current_pos, current_direction)
        if safe_direction:
            return safe_direction
        
        # If no safe direction found, stick with current direction
        return Move(current_direction)
    
    def get_safe_move_towards_passenger(self) -> Move:
        """Get a safe move towards the nearest passenger."""
        if not self.passengers:
            # No passengers, just keep going safely
            train = self.all_trains[self.nickname]
            current_pos = train["position"]
            current_direction = train["direction"]
            
            if self.check_danger_ahead(current_pos, current_direction):
                safe_direction = self.find_safe_direction(current_pos, current_direction)
                if safe_direction:
                    return safe_direction
            
            return Move(current_direction)
        
        train = self.all_trains[self.nickname]
        current_pos = train["position"]
        current_direction = train["direction"]
        
        # Find the nearest passenger
        nearest_passenger = min(self.passengers, 
                               key=lambda p: self.get_distance(current_pos, p["position"]))
        
        # Get best move towards passenger
        best_move = self.get_next_move_towards(current_pos, current_direction, 
                                              nearest_passenger["position"])
        
        # Check if the best move is safe
        if not self.check_danger_ahead(current_pos, best_move.value):
            return best_move
        
        # If not safe, find an alternative safe direction
        safe_direction = self.find_safe_direction(current_pos, current_direction)
        if safe_direction:
            return safe_direction
        
        # If no safe direction found, stick with current direction
        return Move(current_direction)
    
    def get_next_move_towards(self, current_pos: Tuple[int, int], 
                             current_direction: Tuple[int, int], 
                             target_pos: Tuple[int, int]) -> Move:
        """Get the best next move towards a target position."""
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        
        # Filter out the opposite of our current direction (can't make U-turns)
        opposite_dir = (-current_direction[0], -current_direction[1])
        filtered_moves = [move for move in possible_moves 
                         if move.value != opposite_dir]
        
        # If we're currently aligned with the target on one axis, prioritize that direction
        x_diff = target_pos[0] - current_pos[0]
        y_diff = target_pos[1] - current_pos[1]
        
        if x_diff == 0:  # Aligned vertically
            if y_diff < 0 and Move.UP in filtered_moves:
                return Move.UP
            elif y_diff > 0 and Move.DOWN in filtered_moves:
                return Move.DOWN
        
        if y_diff == 0:  # Aligned horizontally
            if x_diff < 0 and Move.LEFT in filtered_moves:
                return Move.LEFT
            elif x_diff > 0 and Move.RIGHT in filtered_moves:
                return Move.RIGHT
        
        # Not aligned, choose the move that gets us closest
        best_move = min(filtered_moves, key=lambda move: 
                       self.get_distance_after_move(current_pos, move.value, target_pos))
        
        return best_move
    
    def get_distance_after_move(self, current_pos: Tuple[int, int], 
                               move_dir: Tuple[int, int], 
                               target_pos: Tuple[int, int]) -> float:
        """Calculate the distance to the target after making a move."""
        new_pos = (current_pos[0] + move_dir[0] * self.cell_size, 
                  current_pos[1] + move_dir[1] * self.cell_size)
        return self.get_distance(new_pos, target_pos) 