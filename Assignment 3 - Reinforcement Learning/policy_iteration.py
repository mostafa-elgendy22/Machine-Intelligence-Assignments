from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
import numpy as np

from helpers.utils import NotImplemented

# This is a class for a generic Policy Iteration agent
class PolicyIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training
    policy: Dict[S, A]
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        # This initial policy will contain the first available action for each state,
        # except for terminal states where the policy should return None.
        self.policy = {
            state: (None if self.mdp.is_terminal(state) else self.mdp.get_actions(state)[0])
            for state in self.mdp.get_states()
        }
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given the utilities for the current policy, compute the new policy
    def update_policy(self):
        #TODO: Complete this function
        # find all the states
        states = self.mdp.get_states()
       
        for state in states:
            # check if the state is terminal
            if self.mdp.is_terminal(state):
                continue
            
            # find all the possible actions
            actions = self.mdp.get_actions(state)
            
            # initialize the max utility and the max action
            max_utility = float('-inf')
            max_action = None

            for action in actions:
                # find all the next states
                next_states = self.mdp.get_successor(state, action)
                
                U = 0
                # calculate the utility of the state
                for next_state, p in next_states.items():
                    U += p * (self.mdp.get_reward(state, action, next_state) + self.discount_factor * self.utilities[next_state])
                
                # update the values of the max utility and the max action
                if U > max_utility:
                    max_utility = U
                    max_action = action

            # update the policy
            self.policy[state] = max_action
    
    # Given the current policy, compute the utilities for this policy
    # Hint: you can use numpy to solve the linear equations. We recommend that you use numpy.linalg.lstsq
    def update_utilities(self):
        #TODO: Complete this function

        # find the number of states in the MDP
        state_size = len(self.mdp.get_states())
        
        # create the matrix A and vector b
        A = np.zeros((state_size, state_size))
        b = np.zeros(state_size)
        
        # find all the states of the MDP
        states = self.mdp.get_states()
        
        # iterate over all the states
        for i, state in enumerate(states):
            # set the diagonal element of the matrix A to 1
            A[i][i] = 1
            
            # if the state is a terminal state, skip the current iteration (i.e. state)
            if self.mdp.is_terminal(state):
                continue

            # find the best action according to the policy
            action = self.policy[state]
            
            # find all the next states
            next_states = self.mdp.get_successor(state, action)
            
            # evaluate the utility of the state
            for next_state, p in next_states.items():
                A[i][states.index(next_state)] -= p * self.discount_factor
                # update the value of the reward
                b[i] += p * self.mdp.get_reward(state, action, next_state)
        
        # solve the linear equations
        x = np.linalg.lstsq(A, b)[0]
        
        # update the utilities
        self.utilities = {state: x[i] for i, state in enumerate(states)}


    
    # Applies a single utility update followed by a single policy update
    # then returns True if the policy has converged and False otherwise
    def update(self) -> bool:
        #TODO: Complete this function

        # save the old policy to compare it with the new one later
        old_policy = self.policy.copy()
        
        # apply single utility update
        self.update_utilities()
        
        # apply single policy update
        self.update_policy()
        
        # return true if the policy has converged (i.e old_policy = new_policy) and false otherwise 
        return old_policy == self.policy


    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update():
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        #TODO: Complete this function
        if self.mdp.is_terminal(state):
            return None
        # return the best action
        return self.policy[state]

    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            policy = {
                self.mdp.format_state(state): (None if action is None else self.mdp.format_action(action)) 
                for state, action in self.policy.items()
            }
            json.dump({
                "utilities": utilities,
                "policy": policy
            }, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in data['utilities'].items()}
            self.policy = {
                self.mdp.parse_state(state): (None if action is None else self.mdp.parse_action(action)) 
                for state, action in data['policy'].items()
            }
