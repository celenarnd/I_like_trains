import logging
import math
from typing import Tuple, List, Dict, Any, Optional
import heapq

from common.base_agent import BaseAgent
from common.move import Move

class Agent(BaseAgent):
    """
    Safety-First Strategist Agent
    
    Avoids potential collisions with other trains.
    Only picks up passengers when there's low risk.
    Navigates around congestion.
    Still tries to score by picking up and dropping off multiple passengers.
    Uses shortest pathfinding for all moves.
    """
    
    def __init__(self, nickname, network, logger="client.agent"):
        super().__init__(nickname, network, logger)
        self.target_passenger = None
        self.pickup_threshold = 3  # Number of passengers to pick up before delivery
        self.going_to_delivery = False
        self.path = []  # Path to current target
        self.safety_distance = 3 * self.cell_size if self.cell_size else 30  # Default safe distance from other trains
        self.danger_zone_radius = 2 * self.cell_size if self.cell_size else 20  # Area to consider high risk
        self.last_risk_assessment = 0  # Time of last risk assessment
        self.risk_assessment_interval = 5  # Reassess risk every 5 moves
    
    def get_move(self) -> Move:
        """Return the next move prioritizing safety."""
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
        
        # Update safety distance based on cell size
        if self.cell_size and self.safety_distance != 3 * self.cell_size:
            self.safety_distance = 3 * self.cell_size
            self.danger_zone_radius = 2 * self.cell_size
        
        # Check for immediate danger and take evasive action if needed
        evasive_move = self.check_for_danger(current_pos, current_direction)
        if evasive_move:
            self.path = []  # Clear path when taking evasive action
            return evasive_move
        
        # If we're in the delivery zone and have wagons, drop them
        if self.is_in_delivery_zone(current_pos) and wagon_count > 0:
            self.going_to_delivery = False
            return Move.DROP
            
        # If we have enough passengers, head to delivery zone
        if wagon_count >= self.pickup_threshold and not self.going_to_delivery:
            self.going_to_delivery = True
            self.path = []
        
        # If we're heading to delivery zone or have passengers, calculate safe path to delivery zone
        if self.going_to_delivery or wagon_count > 0:
            return self.move_towards_delivery_safely(current_pos, current_direction)
            
        # Otherwise, go for the nearest passenger (but only if it's safe)
        return self.move_towards_safe_passenger(current_pos, current_direction)
    
    def check_for_danger(self, current_pos: Tuple[int, int], current_direction: Tuple[int, int]) -> Optional[Move]:
        """Check for immediate collision danger and return evasive move if needed."""
        other_trains = [train for name, train in self.all_trains.items() if name != self.nickname]
        
        # Check for trains in our immediate path
        for train in other_trains:
            train_pos = train["position"]
            train_dir = train["direction"]
            
            # Calculate where we'll be in the next move
            next_pos = (current_pos[0] + current_direction[0] * self.cell_size,
                        current_pos[1] + current_direction[1] * self.cell_size)
            
            # Calculate where the other train will be in the next move
            train_next_pos = (train_pos[0] + train_dir[0] * self.cell_size,
                              train_pos[1] + train_dir[1] * self.cell_size)
            
            # Check if we'll collide
            if next_pos == train_pos or next_pos == train_next_pos:
                return self.get_evasive_move(current_pos, current_direction, train_pos)
            
            # Also check for wagons
            for wagon_pos in train.get("wagons", []):
                if next_pos == wagon_pos:
                    return self.get_evasive_move(current_pos, current_direction, wagon_pos)
        
        return None
    
    def get_evasive_move(self, current_pos: Tuple[int, int], current_direction: Tuple[int, int], 
                        obstacle_pos: Tuple[int, int]) -> Move:
        """Calculate an evasive move to avoid collision."""
        # Try turning left or right
        left_move = Move.turn_left(Move(current_direction))
        right_move = Move.turn_right(Move(current_direction))
        
        # Calculate positions after left and right turns
        left_pos = (current_pos[0] + left_move.value[0] * self.cell_size,
                  current_pos[1] + left_move.value[1] * self.cell_size)
        right_pos = (current_pos[0] + right_move.value[0] * self.cell_size,
                   current_pos[1] + right_move.value[1] * self.cell_size)
        
        # Check if either move would lead to a collision
        left_safe = self.is_position_safe(left_pos)
        right_safe = self.is_position_safe(right_pos)
        
        # Choose the safe turn that takes us away from the obstacle
        if left_safe and right_safe:
            # Both are safe, choose the one that takes us away from the obstacle
            left_distance = self.manhattan_distance(left_pos, obstacle_pos)
            right_distance = self.manhattan_distance(right_pos, obstacle_pos)
            return left_move if left_distance > right_distance else right_move
        elif left_safe:
            return left_move
        elif right_safe:
            return right_move
        
        # If neither is safe, we're in a tight spot - try to find any safe move
        for move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
            if move.value != current_direction and move.value != (-current_direction[0], -current_direction[1]):
                new_pos = (current_pos[0] + move.value[0] * self.cell_size,
                         current_pos[1] + move.value[1] * self.cell_size)
                if self.is_position_safe(new_pos):
                    return move
        
        # If all else fails, continue in current direction (might lead to collision)
        return Move(current_direction)
    
    def is_position_safe(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is safe (not colliding with trains/wagons and within bounds)."""
        # Check if position is within bounds
        if not self.is_valid_position(pos):
            return False
        
        # Check for collision with other trains and their wagons
        for name, train in self.all_trains.items():
            if name == self.nickname:
                continue
                
            if pos == train["position"]:
                return False
                
            for wagon_pos in train.get("wagons", []):
                if pos == wagon_pos:
                    return False
                    
        return True
    
    def move_towards_safe_passenger(self, current_pos: Tuple[int, int], 
                                 current_direction: Tuple[int, int]) -> Move:
        """Move towards a safe passenger, avoiding high-risk areas."""
        # If no passengers, just continue in current direction
        if not self.passengers:
            return Move(current_direction)
            
        # Calculate risk levels for each passenger
        safe_passengers = []
        for passenger in self.passengers:
            passenger_pos = passenger["position"]
            risk_level = self.assess_risk(passenger_pos)
            
            # Only consider low-risk passengers
            if risk_level < 0.5:  # Threshold for "low risk"
                distance = self.manhattan_distance(current_pos, passenger_pos)
                safe_passengers.append((passenger, distance, risk_level))
        
        # Sort by a combination of distance and risk (prioritize close and safe passengers)
        if safe_passengers:
            safe_passengers.sort(key=lambda p: p[1] * (1 + p[2]))
            target_passenger = safe_passengers[0][0]
            target_pos = target_passenger["position"]
        else:
            # If no safe passengers, find a safe area to wait
            return self.find_safe_waiting_move(current_pos, current_direction)
        
        # Calculate path to the safe passenger
        return self.get_next_move_in_path_safely(current_pos, target_pos, current_direction)
    
    def assess_risk(self, position: Tuple[int, int]) -> float:
        """Assess the risk level of a position (0.0 = safe, 1.0 = dangerous)."""
        # Calculate distance to other trains
        risk = 0.0
        for name, train in self.all_trains.items():
            if name == self.nickname:
                continue
                
            train_pos = train["position"]
            distance = self.manhattan_distance(position, train_pos)
            
            # Higher risk for close trains
            if distance < self.danger_zone_radius:
                risk = max(risk, 1.0 - (distance / self.danger_zone_radius))
                
            # Check train's wagons too
            for wagon_pos in train.get("wagons", []):
                wagon_distance = self.manhattan_distance(position, wagon_pos)
                if wagon_distance < self.danger_zone_radius:
                    risk = max(risk, 0.8 * (1.0 - (wagon_distance / self.danger_zone_radius)))
                    
        return risk
    
    def find_safe_waiting_move(self, current_pos: Tuple[int, int], 
                             current_direction: Tuple[int, int]) -> Move:
        """Find a safe place to wait when no safe passengers are available."""
        # Try to find a low-risk area to move to
        lowest_risk = 1.0
        best_move = Move(current_direction)
        
        for move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
            # Skip moves that would cause U-turns
            if self.is_opposite_direction(move.value, current_direction):
                continue
                
            new_pos = (current_pos[0] + move.value[0] * self.cell_size,
                     current_pos[1] + move.value[1] * self.cell_size)
                     
            # Check if position is valid and assess its risk
            if self.is_valid_position(new_pos):
                risk = self.assess_risk(new_pos)
                if risk < lowest_risk:
                    lowest_risk = risk
                    best_move = move
        
        return best_move
    
    def move_towards_delivery_safely(self, current_pos: Tuple[int, int], 
                                  current_direction: Tuple[int, int]) -> Move:
        """Move towards the delivery zone using the safest path."""
        # Calculate the center of the delivery zone
        dz_pos = self.delivery_zone["position"]
        dz_width = self.delivery_zone["width"]
        dz_height = self.delivery_zone["height"]
        
        # Target the center of the delivery zone
        target_pos = (
            dz_pos[0] + dz_width // 2,
            dz_pos[1] + dz_height // 2
        )
        
        # Get next move in path to delivery zone, prioritizing safety
        return self.get_next_move_in_path_safely(current_pos, target_pos, current_direction)
    
    def get_next_move_in_path_safely(self, current_pos: Tuple[int, int], 
                                   target_pos: Tuple[int, int], 
                                   current_direction: Tuple[int, int]) -> Move:
        """Calculate the next move in the path using A* algorithm with safety considerations."""
        # If we're already at the target, no need to move
        if current_pos == target_pos:
            return Move(current_direction)
        
        # If we already have a path, follow it
        if self.path:
            next_pos = self.path[0]
            
            # Check if the next position in our path is still safe
            if not self.is_position_safe(next_pos):
                # Path is no longer safe, recalculate
                self.path = []
            else:
                dx = (next_pos[0] - current_pos[0]) // self.cell_size
                dy = (next_pos[1] - current_pos[1]) // self.cell_size
                
                # If we've reached this position, remove it from the path
                if current_pos == next_pos:
                    self.path.pop(0)
                    return self.get_next_move_in_path_safely(current_pos, target_pos, current_direction)
                
                # Convert the direction to a Move
                for move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
                    if move.value == (dx, dy):
                        # Can't make U-turns
                        if not self.is_opposite_direction(move.value, current_direction):
                            return move
                
                # If we can't make the move, recalculate the path
                self.path = []
        
        # Calculate a new path using A* with safety considerations
        self.path = self.find_safe_path(current_pos, target_pos, current_direction)
        
        # If no path was found, use simple directional movement with safety checks
        if not self.path:
            return self.direct_move_towards_safely(current_pos, target_pos, current_direction)
        
        # Return the next move in the path
        return self.get_next_move_in_path_safely(current_pos, target_pos, current_direction)
    
    def find_safe_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                     current_direction: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a path from start to goal using A* algorithm with safety considerations."""
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
            neighbors = self.get_valid_neighbors_safely(current, current_direction if current == start else None)
            
            for neighbor, risk in neighbors:
                # Skip if neighbor is already processed
                if neighbor in closed_set:
                    continue
                
                # Calculate g_score for this neighbor, including risk penalty
                risk_penalty = risk * self.cell_size * 2  # Higher risk = higher cost
                tentative_g_score = g_score[current] + self.cell_size + risk_penalty
                
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
    
    def get_valid_neighbors_safely(self, pos: Tuple[int, int], 
                                 current_direction: Optional[Tuple[int, int]] = None) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighboring positions with their risk levels."""
        neighbors = []
        
        for move in [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]:
            # Skip if this would be a U-turn
            if current_direction and self.is_opposite_direction(move.value, current_direction):
                continue
                
            # Calculate new position
            new_x = pos[0] + move.value[0] * self.cell_size
            new_y = pos[1] + move.value[1] * self.cell_size
            new_pos = (new_x, new_y)
            
            # Check if valid position
            if self.is_valid_position(new_pos):
                # Calculate risk for this position
                risk = self.assess_risk(new_pos)
                neighbors.append((new_pos, risk))
                
        return neighbors
    
    def direct_move_towards_safely(self, current_pos: Tuple[int, int], 
                                 target_pos: Tuple[int, int], 
                                 current_direction: Tuple[int, int]) -> Move:
        """Make a direct move towards the target position while considering safety."""
        # Get all possible moves (excluding U-turns)
        possible_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
        valid_moves = []
        
        for move in possible_moves:
            # Skip moves that would cause U-turns
            if self.is_opposite_direction(move.value, current_direction):
                continue
                
            # Calculate new position
            new_pos = (current_pos[0] + move.value[0] * self.cell_size,
                     current_pos[1] + move.value[1] * self.cell_size)
                     
            # Check if position is valid
            if self.is_valid_position(new_pos):
                # Calculate risk and distance
                risk = self.assess_risk(new_pos)
                distance = self.manhattan_distance(new_pos, target_pos)
                
                # Add to valid moves with its score (lower is better)
                score = distance * (1 + risk * 2)  # Penalize risky moves
                valid_moves.append((move, score))
        
        if valid_moves:
            # Choose move with lowest score
            valid_moves.sort(key=lambda x: x[1])
            return valid_moves[0][0]
        
        # If no valid moves, continue in current direction
        return Move(current_direction)
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within bounds)."""
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= self.game_width or y < 0 or y >= self.game_height:
            return False
            
        return True
    
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