from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json

from helpers.utils import NotImplemented

# This is a class for a generic Value Iteration agent
class ValueIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training 
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given a state, compute its utility using the bellman equation
    # if the state is terminal, return 0
    def compute_bellman(self, state: S) -> float:
        #TODO: Complete this function
        # if current state is terminal then return 0
        if self.mdp.is_terminal(state):
            return 0

        # find all actions based on current state
        actions = self.mdp.get_actions(state)
        
        # create temp array to store utilities to choose the max utility after this step
        results = []

        for action in actions:
            # find next states (successors) based on the action
            successors = self.mdp.get_successor(state, action)
            
            # initialize the sum to zero
            U = 0
            
            # preform the bellman equation on the successors
            for successor in successors:
                U += successors[successor] * (self.mdp.get_reward(state, action, successor) + self.discount_factor * self.utilities[successor])
            # store in the temp array
            results.append(U)
        
        # return the max utility
        return max(results)
    
    # Applies a single utility update
    # then returns True if the utilities has converged (the maximum utility change is less or equal the tolerance)
    # and False otherwise
    def update(self, tolerance: float = 0) -> bool:
        #TODO: Complete this function

        # create temp array to store the utility changes
        utility_change = []

        # create a new dictionary to use it to update the utilities after iterating over all states
        new_utilities = {}

        for state in self.mdp.get_states():
            # find the utility of the current state
            U = self.compute_bellman(state)
            
            # store the utility change
            utility_change.append(abs(U - self.utilities[state]))
            
            # update the utilities
            new_utilities[state] = U

        # update the utilities
        self.utilities = new_utilities

        # if the max utility change is less than the tolerance then return true
        if max(utility_change) <= tolerance:
            return True
        return False
        


    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None, tolerance: float = 0) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update(tolerance):
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        #TODO: Complete this function
        
        # if current state is terminal then return None
        if self.mdp.is_terminal(state):
            return None

        # get all actions for the current state
        actions = self.mdp.get_actions(state)
        
        # create temp array to store utilities to choose the action that leads to the max utility
        utilities = []
        
        for action in actions:
            # find next states based on action
            successors = self.mdp.get_successor(state, action)
            
            U = 0
            # evaluate the utility of this action
            for successor in successors:
                U += successors[successor] * (self.mdp.get_reward(state, action, successor) + self.discount_factor * self.utilities[successor])
            # store it in the temp array
            utilities.append(U)
        
        # find the arg max action
        max_item = 0
        for i in range(len(utilities)):
            if utilities[i] > utilities[max_item]:
                max_item = i
        return actions[max_item]
    
    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            json.dump(utilities, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            utilities = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in utilities.items()}
