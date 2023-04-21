from typing import Tuple
from game import HeuristicFunction, Game, S, A
from helpers.utils import NotImplemented

#TODO: Import any modules you want to use
import math

# All search functions take a problem, a state, a heuristic function and the maximum search depth.
# If the maximum search depth is -1, then there should be no depth cutoff (The expansion should not stop before reaching a terminal state) 

# All the search functions should return the expected tree value and the best action to take based on the search results

# This is a simple search function that looks 1-step ahead and returns the action that lead to highest heuristic value.
# This algorithm is bad if the heuristic function is weak. That is why we use minimax search to look ahead for many steps.
def greedy(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state)
    
    terminal, values = game.is_terminal(state)
    if terminal: return values[agent], None

    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]
    value, _, action = max((heuristic(game, state, agent), -index, action) for index, (action , state) in enumerate(actions_states))
    return value, action

# Apply Minimax search and return the game tree value and the best action
# Hint: There may be more than one player, and in all the testcases, it is guaranteed that 
# game.get_turn(state) will return 0 (which means it is the turn of the player). All the other players
# (turn > 0) will be enemies. So for any state "s", if the game.get_turn(s) == 0, it should a max node,
# and if it is > 0, it should be a min node. Also remember that game.is_terminal(s), returns the values
# for all the agents. So to get the value for the player (which acts at the max nodes), you need to
# get values[0].
def minimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    terminal, values = game.is_terminal(state)
    agent = game.get_turn(state)

    # check if this state is a terminal state to return the utility
    if terminal: 
        return values[0], None
    
    # check the depth to return the heuristic for the max depth from the current state
    if max_depth == 0: 
        return heuristic(game, state, 0), None

    # if it is MAX
    if agent == 0:
        value = -math.inf
        # for all the successors states we get the max value it can get
        for action in game.get_actions(state):
            successor = game.get_successor(state, action)
            successor_value = minimax(game, successor, heuristic, max_depth - 1)[0]
            if successor_value > value:
                value = successor_value
                best_action = action
    
    # if it is MIN
    else:
        value = math.inf
        # for all the successors states we get the min value it can get
        for action in game.get_actions(state):
            successor = game.get_successor(state, action)
            successor_value = minimax(game, successor, heuristic, max_depth - 1)[0]
            if successor_value < value:
                value = successor_value
                best_action = action

    return value, best_action

# Apply Alpha Beta pruning and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1, alpha: int = -math.inf, beta: int = math.inf) -> Tuple[float, A]:
    terminal, values = game.is_terminal(state)
    agent = game.get_turn(state)

    # check if this state is a terminal state to return the utility
    if terminal: 
        return values[0], None
    
    # check the depth to return the heuristic for the max depth from the current state
    if max_depth == 0: 
        return heuristic(game, state, 0), None

    # if it is MAX
    if agent == 0:
        value = -math.inf
        # for all the successors states we get the max value it can get
        for action in game.get_actions(state):
            successor = game.get_successor(state, action)
            successor_value = alphabeta(game, successor, heuristic, max_depth - 1, alpha, beta)[0]
            if successor_value > value: 
                value = successor_value
                best_action = action
            # check the pruning condtion
            if value >= beta: 
                return value, best_action
            alpha = max(alpha, value)

    # if it is MIN
    else:
        value = math.inf
        # for all the successors states we get the min value it can get
        for action in game.get_actions(state):
            successor = game.get_successor(state, action)
            successor_value = alphabeta(game, successor, heuristic, max_depth - 1, alpha, beta)[0]
            if successor_value <= value: 
                value = successor_value
                best_action = action
            # check the pruning condtion
            if value <= alpha: 
                return value, best_action
            beta = min(beta, value)
            
    return value, best_action


# Apply Alpha Beta pruning with move ordering and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta_with_move_ordering(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1, alpha: int = -math.inf, beta: int = math.inf) -> Tuple[float, A]:
    terminal, values = game.is_terminal(state)
    agent = game.get_turn(state)

    # check if this state is a terminal state to return the utility
    if terminal: 
        return values[0], None
    
    # check the depth to return the heuristic for the max depth from the current state
    if max_depth == 0: 
        return heuristic(game, state, 0), None

    # get all actions with successors
    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]

    # get the heuristic for all the successors
    heuristic_order = [(-heuristic(game, state, 0), index) for index, (_ , state) in enumerate(actions_states)]
    
    # sort based on the agent
    if agent == 0:
        heuristic_order.sort()
    else:
        heuristic_order.sort(reverse=True)
        
    actions_states = [actions_states[ind] for _, ind in heuristic_order]
    
    # if it is MAX
    if agent == 0:
        value = -math.inf
        # for all the successors states we get the max value it can get
        for action, successor in actions_states:
            successor_value = alphabeta_with_move_ordering(game, successor, heuristic, max_depth - 1, alpha, beta)[0]
            if successor_value > value: 
                value = successor_value
                best_action = action
            
            # check the pruning condtion
            if value >= beta: 
                return value, best_action
            alpha = max(alpha, value)
    
    # if it is MIN      
    else:
        value = math.inf
        # for all the successors states we get the min value it can get
        for action, successor in actions_states:
            successor_value = alphabeta_with_move_ordering(game, successor, heuristic, max_depth - 1, alpha, beta)[0]
            if successor_value <= value: 
                value = successor_value
                best_action = action
            
            # check the pruning condtion
            if value <= alpha: 
                return value, best_action
            beta = min(beta, value)
    return value, best_action

# Apply Expectimax search and return the tree value and the best action
# Hint: Read the hint for minimax, but note that the monsters (turn > 0) do not act as min nodes anymore,
# they now act as chance nodes (they act randomly).
def expectimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    terminal, values = game.is_terminal(state)

    agent = game.get_turn(state)

    # check if this state is a termianl state to return the utility
    if terminal: 
        return values[0], None
    
    # check the depth to return the heuristic for the max depth from the current state
    if max_depth == 0: 
        return heuristic(game, state, 0), None

    # if it is MAX
    if agent == 0:
        value = -math.inf
        best_action = None
        # for all the successors states we get the max value it can get
        for action in game.get_actions(state):
            successor = game.get_successor(state, action)
            successor_value = expectimax(game, successor, heuristic, max_depth - 1)[0]
            if successor_value > value:
                value = successor_value
                best_action = action
        return value, best_action
    
    # if it is CHANCE
    else:
        value = 0
        best_action = None
        for action in game.get_actions(state):
            successor = game.get_successor(state, action)
            successor_value = expectimax(game, successor, heuristic, max_depth - 1)[0]
            # accumulate the value with the successor value
            value += successor_value
        # return the average value and the best action
        return value / len(game.get_actions(state)), best_action