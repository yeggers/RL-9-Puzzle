from typing import NamedTuple, Tuple

import numpy as np


class GameDynamicsParameters(NamedTuple):
    """Parameters of the 15-puzzle.

    Parameters:
      N (int): number of rows/columns in the puzzle 
    
    """

    N: int
 

class GameDynamicsState(NamedTuple):
    """State of the 15-puzzle.

    Parameters:
        new_puzzle (np.array): array containing current puzzle
        solution (np.array): solution to the puzzle
        blank_tile_idx (tuple): index of the blank tile
        swapped_tile_idx (tuple): index of the swapped tile
        won (bool): bool specifying whether game has been won 
    """

    new_puzzle: np.array = []
    solution: np.array = []
    blank_tile_idx: tuple = ()
    swapped_tile_idx: tuple = ()
    won: bool = False


class GameDyanmics(): 
    """ 
    Initialize puzzle and perform tile swaps

    Parameters: 
        p (GameDyanmicsParameters): game parameters
      
    """
    def __init__(self, p):
       self.p = p
        
    def find_blank_index(self, new_puzzle): 
        """ 
        Find index of the blank tile

        Parameters: 
            new_puzzle (np.array): input puzzle
        
        """      
        blank_index = np.where(new_puzzle == 0)
        
        return (int(blank_index[0]), int(blank_index[1]))

    
    def check_if_solvable_b(self, new_puzzle):
        #Initiate number of inversions
        inversions = 0
        
        #Compute number of inversions
        for i in range(len(new_puzzle)):
            inversions += sum(j < new_puzzle[i] and j!=0 for j in new_puzzle[i:])
            test0 = sum(j < new_puzzle[i] and j!=0 for j in new_puzzle[i:])

        #Determine row of blank 
        blank_tile_index = self.find_blank_index(new_puzzle.reshape(self.p.N, self.p.N))
        blank_tile_row = blank_tile_index[0]

        #Check if solvable
        if (self.p.N%2 == 0 and ((inversions%2 == 0 and blank_tile_row%2 != 0) or (inversions%2 != 0 and blank_tile_row%2 == 0))) or \
        (self.p.N%2!= 0 and inversions%2 == 0):
            
            return True, blank_tile_index
        
        return False, blank_tile_index
    

   
    def check_if_solvable(self, new_puzzle):
        """ 
        Check if the created puzzle can be solved using method described in https://www-jstor-org.tudelft.idm.oclc.org/stable/2369492?seq=6#metadata_info_tab_contents.
        
        
        If N is odd puzzle is solvable if number of inversions is even in input state

        If N is even puzzle is solvable if:
            
            - the blank is on an even row counting from the bottom and number of inversions is odd.
            - the blank is on an odd row counting from the bottom and number of inversions is even.

        Definition inversion: A pair of tiles (a,b) form an inversion if a appears before b but a<b in the 1D-array of the puzzle.
    

        Parameters: 
            new_puzzle (np.array): input puzzle 
        
        """
        #Initiate puzzle as unsolvable
        solvable = False 

        #Make new puzzles until solvable one is found
        while solvable == False:

            solvable, blank_tile_index = self.check_if_solvable_b(new_puzzle)
            
            if solvable == False:
                np.random.shuffle(new_puzzle)


        return new_puzzle, blank_tile_index

    
    
    def make_new_puzzle(self):
        """ 
        Set up a new puzzle. Arrays are kept 1-dimensional until they are returned since it is easier to check whether they are solvable this way.

        Parameters: 
        
        """
        #Define solution to the puzzle 
        solution = np.append(np.arange(1, self.p.N**2), 0)

        #Create new puzzle 
        new_puzzle = np.random.choice(solution, self.p.N**2, replace = False)

        #Check if created puzzle is solvable and if not make new ones until solvable one is found
        new_puzzle, blank_index = self.check_if_solvable(new_puzzle)

        #Convert 1D arrays to NxN arrays
        return new_puzzle.reshape(self.p.N, self.p.N), solution.reshape(self.p.N, self.p.N), blank_index


    def swap_tiles(self,new_puzzle, blank_tile_idx, swapped_tile_idx):

        """
            Swap position of blank and selected tile

            Parameters: 
               new_puzzle (np.array): puzzle in array form
               blank_tile_idx (tuple): index of blank tile
               swapped_tile_idx (tuple): index of swapped tile


        """
        #swap selected and blank tile
        new_puzzle[blank_tile_idx] = new_puzzle[swapped_tile_idx]
        new_puzzle[swapped_tile_idx] = 0

        return new_puzzle


    def check_won(self, solution, new_puzzle):

        """
            Check if updated puzzle corresponds to correct solution
            
            Parameters: 
                solution (np.array): solution to the game
                new_puzzle (np.array): puzzle in array form

        """

        if np.array_equal(solution, new_puzzle):
            return True
            
        else:
            return False
