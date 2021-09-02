from typing import NamedTuple, Tuple
#import torch 
import numpy as np
import game.game_dyanmics as game_dynamics

class UpdateGame():
    """ 
        Update the puzzle based on the chosen tile (determined either manually by player, by trained RL or by training RL) and compare to correct solution

        Parameters: 
            N (int): number of rows/columns in puzzle 
            new_puzzle (np.array): previous state of the puzzle
            solution (np.array): solution to the puzzle
            swapped_tile_idx (tuple): index of tile to be swapped
   
    """
    def __init__(self, N, new_puzzle, solution, swapped_tile_idx):
     
        self.N = N
        self.new_puzzle = new_puzzle
        self.solution = solution 
        self.swapped_tile_idx = swapped_tile_idx
        self.blank_idx = game_dynamics.NewPuzzle(N).find_blank_index(new_puzzle)
        
        
   
    def swap_tiles(
        self
    ) -> Tuple[np.array, tuple]:
        
        """
            Swap position of blank and selected tile

            Parameters: 
    
        """

        #swap selected and blank tile
        self.new_puzzle[self.blank_idx] = self.new_puzzle[self.swapped_tile_idx]
        self.new_puzzle[self.swapped_tile_idx] = 0

        #Update index of blank tile
        self.blank_idx = self.swapped_tile_idx

        return self.new_puzzle, self.blank_idx

    
    def check_won(
        self, 
        updated_puzzle: np.array
    )-> bool:

        """
            Check if updated puzzle corresponds to correct solution
            
            Parameters: 
    
        """

        if np.array_equal(self.solution, updated_puzzle):
            return True
        else:
            return False