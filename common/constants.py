# The standard tick rate used as a reference for game timing calculations.
# This value ensures consistent game duration regardless of the actual tick_rate.
# - The game's internal time calculations are always based on this reference value
# - When tick_rate equals reference_tick_rate, the game runs at normal speed
# - When tick_rate is higher/lower than reference_tick_rate, the game runs faster/slower
# - In grading mode, tick_rate is set to a very high value while reference_tick_rate stays the same,
#   causing the game to run much faster while maintaining consistent game logic
REFERENCE_TICK_RATE: int = 60
