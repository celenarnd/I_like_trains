import logging
import math
from typing import Tuple, List, Dict, Any

from common.base_agent import BaseAgent
from common.move import Move

class Agent(BaseAgent):
    """
    Greedy Agent
    
    Always goes for the nearest available passenger and drops them off as fast as possible.
    Prioritizes scoring over survival.
    """
    
    def get_move(self) -> Move:
        """Return the best move for the agent based on the greedy strategy."""
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
            
        # If we have wagons, head to delivery zone
        if train.get("wagons", []):
            return self.get_move_towards_delivery()
            
        # No wagons, go to nearest passenger
        return self.get_move_towards_nearest_passenger()
    
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
    
    def get_move_towards_delivery(self) -> Move:
        """Get the move that takes us towards the delivery zone."""
        train = self.all_trains[self.nickname]
        current_pos = train["position"]
        current_direction = train["direction"]
        
        # Calculate the center of the delivery zone
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        dz_center = (dz_pos[0] + dz_width // 2, dz_pos[1] + dz_height // 2)
        
        return self.get_next_move_towards(current_pos, current_direction, dz_center)
    
    def get_move_towards_nearest_passenger(self) -> Move:
        """Get the move that takes us towards the nearest passenger."""
        if not self.passengers:
            # No passengers, just keep going
            return Move(self.all_trains[self.nickname]["direction"])
        
        train = self.all_trains[self.nickname]
        current_pos = train["position"]
        current_direction = train["direction"]
        
        # Find the nearest passenger
        nearest_passenger = min(self.passengers, 
                               key=lambda p: self.get_distance(current_pos, p["position"]))
        
        return self.get_next_move_towards(current_pos, current_direction, 
                                         nearest_passenger["position"])
    
    def get_next_move_towards(self, current_pos: Tuple[int, int], 
                             current_direction: Tuple[int, int], 
                             target_pos: Tuple[int, int]) -> Move:
        """
        Get the best next move towards a target position.
        This is a greedy approach that tries to get closer to the target.
        """
        # Calculate distances for each possible move
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