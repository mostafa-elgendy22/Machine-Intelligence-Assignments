from typing import Any, Dict, List, Optional
from CSP import Assignment, BinaryConstraint, Problem, UnaryConstraint
from helpers.utils import NotImplemented
import copy

# This function should apply 1-Consistency to the problem.
# In other words, it should modify the domains to only include values that satisfy their variables' unary constraints.
# Then all unary constraints should be removed from the problem (they are no longer needed).
# The function should return False if any domain becomes empty. Otherwise, it should return True.
def one_consistency(problem: Problem) -> bool:
    #TODO: Write this function
    domains = problem.domains
    constraints = problem.constraints
    
    # loop over all the constraints of the problem
    for constraint in constraints:
        # choose the unary constraints only
        if isinstance(constraint, UnaryConstraint):
            variable = constraint.variable
            condition = constraint.condition
            # get the domain of the variable
            domain = domains[variable]
            for value in domain.copy():
                # remove the value from the domain if it doesn't satisfy the condition
                if not condition(value):
                    domain.remove(value)
            if len(domain) == 0:
                return False
    return True

# This function should implement forward checking
# The function is given the problem, the variable that has been assigned and its assigned value and the domains of the unassigned values
# The function should return False if it is impossible to solve the problem after the given assignment, and True otherwise.
# In general, the function should do the following:
#   - For each binary constraints that involve the assigned variable:
#       - Get the other involved variable.
#       - If the other variable has no domain (in other words, it is already assigned), skip this constraint.
#       - Update the other variable's domain to only include the values that satisfy the binary constraint with the assigned variable.
#   - If any variable's domain becomes empty, return False. Otherwise, return True.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def forward_checking(problem: Problem, assigned_variable: str, assigned_value: Any, domains: Dict[str, set]) -> bool:
    #TODO: Write this function
    constraints = problem.constraints
    # loop over all the constraints of the problem
    for constraint in constraints:
        # choose the binary constraints only
        if isinstance(constraint, BinaryConstraint):
            
            # check if the assigned variable is involved in the constraint
            if assigned_variable in constraint.variables:
                # get the other variable
                other_variable = constraint.variables[0] if constraint.variables[1] == assigned_variable else constraint.variables[1]
                
                if other_variable in domains:
                    # get the domain of the other variable
                    domain = domains[other_variable]
            
                    # remove the values that don't satisfy the constraint
                    for value in domain.copy():
                        if not constraint.condition(assigned_value, value):
                            domain.remove(value)
                    
                    if len(domain) == 0:
                        return False
    return True

# This function should return the domain of the given variable order based on the "least restraining value" heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# Generally, this function is very similar to the forward checking function, but it differs as follows:
#   - You are not given a value for the given variable, since you should do the process for every value in the variable's
#     domain to see how much it will restrain the neigbors domain
#   - Here, you do not modify the given domains. But you can create and modify a copy.
# IMPORTANT: If multiple values have the same priority given the "least restraining value" heuristic, 
#            order them in ascending order (from the lowest to the highest value).
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def least_restraining_values(problem: Problem, variable_to_assign: str, domains: Dict[str, set]) -> List[Any]:
    # check if the domains are empty
    if len(domains) == 0: 
        return []
    
    # get the binary constraints
    binary_constraints = [constraint for constraint in problem.constraints if isinstance(constraint, BinaryConstraint)]
    
    # create a list that stores the values and their priority
    value_priority = []
    
    if variable_to_assign in domains:
        # get the domain of the variable to be assigned
        domain = domains[variable_to_assign]
        values = list(domain)
        
        # sort the values in ascending order
        values.sort()
        
        # loop over the values in the domain
        for value in values:
            
            # initialize the priority to 0
            priority = 0

            # loop over binary constraints
            for constraint in binary_constraints:
                # check if the variable to be assigned is involved in the constraint
                if variable_to_assign in constraint.variables:
                    # get the other variable
                    other_variable = constraint.get_other(variable_to_assign)
                    
                    if other_variable in domains:
                        # get the domain of the other variable
                        other_variable_domain = domains[other_variable]

                        # loop over the values in the domain of the other variable
                        for other_variable_value in other_variable_domain:
                            # if the value satisfies the constraint, decrease the priority
                            if constraint.condition(value, other_variable_value):
                                priority = priority - 1
            
            # add the value and its priority to the list
            value_priority.append((value, priority))
    
    # sort the list based on the priority
    value_priority.sort(key=lambda x: x[1])
    
    # return the values in ascending order of priority
    return [value for value, _ in value_priority]

# This function should return the variable that should be picked based on the MRV heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
# IMPORTANT: If multiple variables have the same priority given the MRV heuristic, 
#            order them in the same order in which they appear in "problem.variables".
def minimum_remaining_values(problem: Problem, domains: Dict[str, set]) -> str:
    #TODO: Write this function
    variables = problem.variables
    
    # initialize the minimum domain to infinity and the minimum variable to None
    min_domain = float('inf')
    min_variable = None
    
    # loop over all the variables
    for variable in variables:
        if variable in domains:
            # get the domain of the variable
            domain = domains[variable]
            # check if the domain length is smaller than the minimum domain length
            if len(domain) < min_domain:
                # update the minimum domain and the minimum variable
                min_domain = len(domain)
                min_variable = variable
    return min_variable

# This function should solve CSP problems using backtracking search with forward checking.
# The variable ordering should be decided by the MRV heuristic.
# The value ordering should be decided by the "least restraining value" heurisitc.
# Unary constraints should be handled using 1-Consistency before starting the backtracking search.
# This function should return the first solution it finds (a complete assignment that satisfies the problem constraints).
# If no solution was found, it should return None.
# IMPORTANT: To get the correct result for the explored nodes, you should check if the assignment is complete only once using "problem.is_complete"
#            for every assignment including the initial empty assignment, EXCEPT for the assignments pruned by the forward checking.
#            Also, if 1-Consistency deems the whole problem unsolvable, you shouldn't call "problem.is_complete" at all.

def rec_solve(problem: Problem, domains: Dict, assignment: Assignment = {}) -> Optional[Assignment]:
    # Return None if the problem is not 1-consistent
    if not one_consistency(problem):
        return None

    # Return the assignment if it is complete
    if problem.is_complete(assignment):
        return assignment

    # Choose the variable with the minimum remaining values to be assigned
    x = minimum_remaining_values(problem, domains)
    
    # Choose the domain of the variable based on the "least restraining value" heuristic
    values = least_restraining_values(problem, x, domains)

    for value in values:
        # Use deepcopy to copy the value not the reference of the assignment and the domains
        assignment_copy = copy.deepcopy(assignment)
        assignment_copy[x] = value

        domains_copy = copy.deepcopy(domains)
        del domains_copy[x]

        # Check if the assignment is consistent with the constraints    
        if forward_checking(problem, x, value, domains_copy):
            # Call the recursive function
            result = rec_solve(problem, domains_copy, assignment_copy)
            
            # Return the result assignment if it is not None
            if result is not None:
                return result
    return None

def solve(problem: Problem) -> Optional[Assignment]:
    #TODO: Write this function
    # Call the recursive function
    return rec_solve(problem, problem.domains)
    