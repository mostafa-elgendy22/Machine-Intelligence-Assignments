from typing import Any, Set, Tuple
from grid import Grid
import utils

def locate(grid: Grid, item: Any) -> Set[Tuple[int,int]]:
    '''
    This function takes a 2D grid and an item
    It should return a list of (x, y) coordinates that specify the locations that contain the given item
    To know how to use the Grid class, see the file "grid.py"  
    '''
    #TODO: ADD YOUR CODE HERE
    locations: Set[Tuple[int,int]] = set()

    width = grid.width
    height = grid.height

    for i in range(width):
        for j in range(height):
            if grid.__getitem__((i, j)) == item:
                locations.add((i, j))

    return locations