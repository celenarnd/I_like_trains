import random
import logging
import math


# Use the logger configured in server.py
logger = logging.getLogger("server.delivery_zone")


class DeliveryZone:
    """
    Represents the area where passengers must be dropped off in order to earn points.

    DeliveryZones are placed randomly. Their size depends on the number of players.
    """

    def __init__(self, game_width, game_height, cell_size, nb_players, random_gen=None):
        self.random = random_gen if random_gen is not None else random

        # Calculate a factor based on square root for slower growth
        # Ensure nb_players is positive. Use sqrt + small linear term.
        player_factor = (math.isqrt(nb_players) ) if nb_players > 0 else 0

        # Both dimensions should depend on player factor
        width_with_factor = player_factor
        height_with_factor = player_factor

        # Randomly choose which dimension gets an extra boost
        random_increased_dimension = self.random.choice(["width", "height"])
        
        # Apply cell size scaling to final dimensions
        self.width = cell_size * (
            width_with_factor + player_factor if random_increased_dimension == "width" else width_with_factor
        )
        self.height = cell_size * (
            height_with_factor + player_factor if random_increased_dimension == "height" else height_with_factor
        )
        
        # Calculate and clamp the upper bound for x
        max_x_offset = game_width // cell_size - 1 - self.width // cell_size
        upper_bound_x = max(0, max_x_offset)
        self.x = cell_size * self.random.randint(0, upper_bound_x)
        
        # Calculate and clamp the upper bound for y
        max_y_offset = (
            (game_height // cell_size - 1 - self.height // cell_size)
        )
        # Ensure the upper bound is not negative
        upper_bound_y = max(0, max_y_offset)

        self.y = cell_size * self.random.randint(0, upper_bound_y)

        logger.debug(f"Delivery zone bounds: ({self.x}, {self.y}, {self.x + self.width}, {self.y + self.height})")


    def contains(self, position):
        x, y = position
        return (
            x >= self.x
            and x < self.x + self.width
            and y >= self.y
            and y < self.y + self.height
        )

    def to_dict(self):
        return {
            "height": self.height,
            "width": self.width,
            "position": (self.x, self.y),
        }
