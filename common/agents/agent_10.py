import logging
import math
from typing import Tuple, List, Dict, Any, Optional

from common.base_agent import BaseAgent
from common.move import Move

class Agent(BaseAgent):
    """
    Greedy Agent
    
    Always targets the nearest available passenger and drops them off as fast as possible.
    Prioritizes scoring over everything else.
    Picks up multiple passengers before returning to delivery zone.
    Can move diagonally by alternating between horizontal and vertical moves.
    """
    
    def __init__(self, nickname, network, logger="client.agent"):
        super().__init__(nickname, network, logger)
        self.last_move = None  # Track last move for diagonal movement
        self.pickup_threshold = 3  # Number of passengers to pick up before delivery
        self.target_passenger = None  # Current passenger being targeted
        self.going_to_delivery = False  # Whether we're heading to delivery zone
        self.path = []  # Path to current target
        
    def get_move(self) -> Move:
        """Return the next move for the greedy agent."""
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
        
        # If we're in the delivery zone and have wagons, drop them
        if self.is_in_delivery_zone(current_pos) and wagon_count > 0:
            self.going_to_delivery = False
            return Move.DROP
            
        # If we have enough passengers, head to delivery zone
        if wagon_count >= self.pickup_threshold and not self.going_to_delivery:
            self.going_to_delivery = True
            self.target_passenger = None
            self.path = []
            
        # Check if an adjacent passenger can be picked up (opportunistic pickup)
        pickup_move = self.check_adjacent_passenger(current_pos, current_direction)
        if pickup_move and wagon_count < self.pickup_threshold:
            self.last_move = pickup_move
            return pickup_move
            
        # Determine target (passenger or delivery zone)
        if self.going_to_delivery:
            return self.move_towards_delivery(current_pos, current_direction)
        else:
            return self.move_towards_nearest_passenger(current_pos, current_direction)
            
    def move_towards_nearest_passenger(self, current_pos: Tuple[int, int], 
                                     current_direction: Tuple[int, int]) -> Move:
        """Move towards the nearest valuable passenger."""
        # If no passengers, just continue in current direction
        if not self.passengers:
            self.last_move = Move(current_direction)
            return Move(current_direction)
            
        # Find nearest passenger if we don't have a target or our target is gone
        if not self.target_passenger or self.target_passenger not in self.passengers:
            self.target_passenger = self.find_best_passenger(current_pos)
            
        # Get the passenger's position
        passenger_pos = self.target_passenger["position"]
        
        # Attempt diagonal movement towards the passenger
        diagonal_move = self.move_diagonally(current_pos, passenger_pos, current_direction)
        if diagonal_move:
            return diagonal_move
            
        # If diagonal movement not possible, use direct movement
        return self.direct_move_towards(current_pos, passenger_pos, current_direction)
        
    def move_towards_delivery(self, current_pos: Tuple[int, int], 
                            current_direction: Tuple[int, int]) -> Move:
        """Move towards the delivery zone."""
        # Get delivery zone position
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        
        # Calculate center position of delivery zone
        delivery_center = (
            dz_pos[0] + dz_width // 2,
            dz_pos[1] + dz_height // 2
        )
        
        # Attempt diagonal movement towards the delivery zone
        diagonal_move = self.move_diagonally(current_pos, delivery_center, current_direction)
        if diagonal_move:
            return diagonal_move
            
        # If diagonal movement not possible, use direct movement
        return self.direct_move_towards(current_pos, delivery_center, current_direction)
    
    def find_best_passenger(self, current_pos: Tuple[int, int]) -> Dict:
        """Find the nearest valuable passenger."""
        # Calculate distance and value scores for all passengers
        scored_passengers = []
        for passenger in self.passengers:
            distance = self.get_distance(current_pos, passenger["position"])
            value = passenger["value"]
            # Score formula: higher values and shorter distances are better
            score = value / max(1, distance / self.cell_size)
            scored_passengers.append((passenger, score))
            
        # Sort by score (higher is better)
        scored_passengers.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest scoring passenger
        return scored_passengers[0][0]
    
    def move_diagonally(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                       current_direction: Tuple[int, int]) -> Optional[Move]:
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
            
        if not valid_moves:
            return None
            
        # For diagonal movement, alternate between horizontal and vertical
        if self.last_move:
            # If last move was horizontal, try vertical next (if possible)
            if self.last_move in [Move.LEFT, Move.RIGHT]:
                vertical_moves = [move for move in valid_moves if move in [Move.UP, Move.DOWN]]
                if vertical_moves:
                    self.last_move = vertical_moves[0]
                    return vertical_moves[0]
            # If last move was vertical, try horizontal next (if possible)
            else:
                horizontal_moves = [move for move in valid_moves if move in [Move.LEFT, Move.RIGHT]]
                if horizontal_moves:
                    self.last_move = horizontal_moves[0]
                    return horizontal_moves[0]
        
        # If no alternating move is possible or no last move, just pick the best direction
        best_move = self.get_best_move(current_pos, target_pos, valid_moves)
        self.last_move = best_move
        return best_move
    
    def direct_move_towards(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                          current_direction: Tuple[int, int]) -> Move:
        """Make a direct move towards the target position."""
        # Calculate the direction to the target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Get all possible moves (excluding U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        opposite_dir = (-current_direction[0], -current_direction[1])
        valid_moves = [move for move in possible_moves if move.value != opposite_dir]
        
        # Calculate the best move
        best_move = self.get_best_move(current_pos, target_pos, valid_moves)
        self.last_move = best_move
        return best_move
    
    def get_best_move(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                     valid_moves: List[Move]) -> Move:
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
        return best_move if best_move else valid_moves[0] if valid_moves else Move.RIGHT
    
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