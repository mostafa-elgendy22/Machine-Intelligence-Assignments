from dungeon import DungeonProblem, DungeonState
from mathutils import Direction, Point, euclidean_distance, manhattan_distance
from helpers import utils

# This heuristic returns the distance between the player and the exit as an estimate for the path cost
# While it is consistent, it does a bad job at estimating the actual cost thus the search will explore a lot of nodes before finding a goal
def weak_heuristic(problem: DungeonProblem, state: DungeonState):
    return euclidean_distance(state.player, problem.layout.exit)

#TODO: Import any modules and write any functions you want to use
import math
    

def strong_heuristic(problem: DungeonProblem, state: DungeonState) -> float:
    # Go to the exit, if there are no more coins are remaining
    if not state.remaining_coins:
        return manhattan_distance(state.player, problem.layout.exit)
    
    # Initialize the minimum distance to infinity
    min_distance = math.inf
    
    for coin in state.remaining_coins:
        # Go to the coin, if the coin is close to the exit
        min_distance = min(min_distance, manhattan_distance(coin, problem.layout.exit) + manhattan_distance(coin, state.player))
    return min_distance