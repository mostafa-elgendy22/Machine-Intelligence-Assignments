from problem import HeuristicFunction, Problem, S, A, Solution
from collections import deque
from helpers import utils

#TODO: Import any modules you want to use

# All search functions take a problem and a state
# If it is an informed search function, it will also receive a heuristic function
# S and A are used for generic typing where S represents the state type and A represents the action type

# All the search functions should return one of two possible type:
# 1. A list of actions which represent the path from the initial state to the final state
# 2. None if there is no solution

def BreadthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    
    # A double-end queue which holds the states to be expanded (FIFO queue)
    # It initially contains the initial state 
    frontier = deque([initial_state])

    # A set which holds all the previously explored states
    explored_set = set({})

    # A dictionary (hash table) to the store the path from the initial state to the current state
    actions = dict({initial_state: list([])})
    
    while len(frontier):
        # The node to be expanded (the first node of the frontier)
        node = frontier.popleft()
        
        # Add the node to be expanded to the explored set
        explored_set.add(node)
        
        # If the curent node's state is goal state then return the sequence of actions to 
        # go from the initial state to the current (goal) state
        if problem.is_goal(node):
            return actions[node]
        
        
        # Loop over all the possible actions of the current state
        for action in problem.get_actions(node):
            child_node = problem.get_successor(node, action)
            
            # Check if the child node is not in the frontier and also not in the explored set
            # to prevent expansion of the same node more than one time
            if child_node not in frontier and child_node not in explored_set:
                frontier.append(child_node)

                # Update the path of the child node so that it contains the actions of the parent
                # node plus the action between the parent node and the child node
                child_node_actions = actions[node].copy()
                child_node_actions.append(action)
                actions[child_node] = child_node_actions
                
    # Return none if there is no found solution
    return None

def DepthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    # A double-end queue which holds the states to be expanded (FIFO queue)
    # It initially contains the initial state 
    frontier = deque([initial_state])

    # A set which holds all the previously explored states
    explored_set = set({})

    # A dictionary (hash table) to the store the path from the initial state to the current state
    actions = dict({initial_state: list([])})
    
    while len(frontier):
        # The node to be expanded (the last node of the frontier)
        node = frontier.pop()
        
        # Add the node to be expanded to the explored set
        explored_set.add(node)
        
        # If the curent node's state is goal state then return the sequence of actions to 
        # go from the initial state to the current (goal) state
        if problem.is_goal(node):
            return actions[node]
        
        
        # Loop over all the possible actions of the current state
        for action in problem.get_actions(node):
            child_node = problem.get_successor(node, action)
            
            # Check if the child node is not in the frontier and also not in the explored set
            # to prevent expansion of the same node more than one time
            if child_node not in frontier and child_node not in explored_set:
                frontier.append(child_node)

                # Update the path of the child node so that it contains the actions of the parent
                # node plus the action between the parent node and the child node
                child_node_actions = actions[node].copy()
                child_node_actions.append(action)
                actions[child_node] = child_node_actions
                
    # Return none if there is no found solution
    return None
    
def UniformCostSearch(problem: Problem[S, A], initial_state: S) -> Solution:

    # A dictionary (hash table) that holds the states to be expanded (key), and the path cost (value)
    frontier = dict({initial_state: 0})
    
    # A set which holds all the previously explored states
    explored_set = set({})
    
    # A dictionary (hash table) to the store the path from the initial state to the current state
    # The key is the state, the value is a list of actions from initial state to current state
    actions = dict({initial_state: list([])})
    
    while len(frontier):
        
        # The node to be expanded (the one with the lowest path cost)
        node = list(frontier.keys())[0]
        node_cost = frontier[node]
        
        # Remove the node form the frontier
        frontier.pop(node)
        
        # Add the node to be expanded to the explored set
        explored_set.add(node)
        
        # If the curent node's state is goal state then return the sequence of actions to 
        # go from the initial state to the current (goal) state
        if problem.is_goal(node):
            return actions[node]
        
        # Loop over all the possible actions of the current state
        for action in problem.get_actions(node):
            # Evaluate the action cost
            action_cost = node_cost + problem.get_cost(node, action)
            
            child_node = problem.get_successor(node, action)
            
            # Check if the child node is not in the frontier and also not in the explored set
            # to prevent expansion of the same node more than one time
            # Or the node exists in the frontier with larger path cost
            if (child_node not in frontier and child_node not in explored_set) \
            or (child_node in frontier and action_cost < frontier[child_node]):
                # Add the child node to the frontier
                frontier[child_node] = action_cost
                
                # Sort the frontier according to the path cost after adding the child node
                # Reference: https://realpython.com/sort-python-dictionary/
                frontier = dict(sorted(frontier.items(), key = lambda item: item[1]))
                
                # Update the path of the child node so that it contains the actions of the parent
                # node plus the action between the parent node and the child node
                child_node_actions = actions[node].copy()
                child_node_actions.append(action)
                actions[child_node] = child_node_actions
                
    # Return none if there is no found solution
    return None

def AStarSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    # A dictionary (hash table) that holds the states to be expanded (key), and the path cost (value)
    frontier = dict({initial_state: 0})
    
    # A set which holds all the previously explored states
    explored_set = set({})
    
    # A dictionary (hash table) to the store the path from the initial state to the current state
    # The key is the state, the value is a list of actions from initial state to current state
    actions = dict({initial_state: list([])})
    
    while len(frontier):
        # The node to be expanded (the one with the lowest path cost)
        node = list(frontier.keys())[0]
        node_cost = frontier[node]
        
        # Remove the node form the frontier
        frontier.pop(node)
        
        # Add the node to be expanded to the explored set
        explored_set.add(node)
        
        # If the curent node's state is goal state then return the sequence of actions to 
        # go from the initial state to the current (goal) state
        if problem.is_goal(node):
            return actions[node]
        
        # Loop over all the possible actions of the current state
        for action in problem.get_actions(node):
            # Evaluate the action cost
            action_cost = node_cost + problem.get_cost(node, action)
            
            child_node = problem.get_successor(node, action)
            
            # Check if the child node is not in the frontier and also not in the explored set
            # to prevent expansion of the same node more than one time
            # Or the node exists in the frontier with larger path cost
            if (child_node not in frontier and child_node not in explored_set) \
            or (child_node in frontier and action_cost < frontier[child_node]):
                
                frontier[child_node] = action_cost
                
                # Sort the frontier according to the path cost after adding the child node
                # Reference: https://realpython.com/sort-python-dictionary/
                frontier = dict(sorted(frontier.items(), 
                                       key = lambda k: k[1] + heuristic(problem, k[0])))
                
                # Update the path of the child node so that it contains the actions of the parent
                # node plus the action between the parent node and the child node
                child_node_actions = actions[node].copy()
                child_node_actions.append(action)
                actions[child_node] = child_node_actions
                
    # Return none if there is no found solution
    return None

def BestFirstSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    
    # A dictionary (hash table) that holds the states to be expanded (key), and the path cost (value)
    frontier = dict({initial_state: 0})
    
    # A set which holds all the previously explored states
    explored_set = set({})
    
    # A dictionary (hash table) to the store the path from the initial state to the current state
    # The key is the state, the value is a list of actions from initial state to current state
    actions = dict({initial_state: list([])})
    
    while len(frontier):
        # The node to be expanded (the one with the lowest path cost)
        node = list(frontier.keys())[0]
        
        # Remove the node form the frontier
        frontier.pop(node)
        
        # Add the node to be expanded to the explored set
        explored_set.add(node)
        
        # If the curent node's state is goal state then return the sequence of actions to 
        # go from the initial state to the current (goal) state
        if problem.is_goal(node):
            return actions[node]

        # Loop over all the possible actions of the current state
        for action in problem.get_actions(node):
            child_node = problem.get_successor(node, action)
            
            # Evaluate the action cost
            action_cost = heuristic(problem, child_node)
            
            # Check if the child node is not in the frontier and also not in the explored set
            # to prevent expansion of the same node more than one time
            # Or the node exists in the frontier with larger path cost
            if (child_node not in frontier and child_node not in explored_set) \
            or (child_node in frontier and action_cost < frontier[child_node]):
                frontier[child_node] = action_cost
                
                # Sort the frontier according to the path cost after adding the child node
                # Reference: https://realpython.com/sort-python-dictionary/
                frontier = dict(sorted(frontier.items(), key = lambda k: k[1]))
                
                # Update the path of the child node so that it contains the actions of the parent
                # node plus the action between the parent node and the child node
                child_node_actions = actions[node].copy()
                child_node_actions.append(action)
                actions[child_node] = child_node_actions
                
    # Return none if there is no found solution
    return None