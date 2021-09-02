
import matplotlib.pyplot as plt
import numpy as np
import pygame

import game.game_dyanmics as game_dyanmics
from game.game_dyanmics import GameDynamicsState
from game.visualize_game import VisualizeGameState


def check_next_blank(blank_tile_idx, swapped_tile_idx):
    """
    Check if clicked tile is next to the blank tile

        Parameters: 
            state_GD (GameDynamicsState): state of the game

    """
    #Determine indices which are next to blank tile 
    next_blank_idx = np.array(blank_tile_idx) + np.array([[-1, 0], [1,0], [0, -1], [0,1]])

    #Check if clicked tile is next to blank tile 
    if np.any(np.all(swapped_tile_idx == next_blank_idx, axis=1)):
        return True 
    
    else:
        return False

def check_if_over(pos_cursor_x, pos_cursor_y, tile):
    """
    Check if cursor is over specific tile 

        Parameters: 
            pos_cursor_x (tuple): x-position of cursor
            pos_cursor_y (tuple): y-position of cursor
            tile (Tile): tile to be evaluated

    """
    
    if (pos_cursor_x in tile.xrange) and (pos_cursor_y in tile.yrange):
            return True  
    return False

def determine_swapped_tile_idx(GameDynamics, VisualizeGame, state_GD, state_VG, event):
    """ 
    Determine which tile is supposed to be swapped/ which tile is clicked

        Parameters:
            GameDynamics (GameDyanmics): instance of GameDyanmics class
            VisualizeGame (VisualizeGame): instance of VisualizeGame class
            state_GD (GameDynamicsState): state of the game
            state_VG (VisualizeState): state of the visualization
            event (pygame.event): event within the game 
            won (bool): bool which describes whether or not game has been won 
        
    """    
    #Determine position of cursor 
    pos_cursor = pygame.mouse.get_pos()

    #Extract states
    new_puzzle, solution, blank_tile_idx, swapped_tile_idx, won = state_GD
    tiles, win = state_VG
    
    
    #Loop through all tiles 
    for index, tile in np.ndenumerate(state_VG.tiles): 
        #Check if mouse buton is pressed
        if event.type == pygame.MOUSEBUTTONDOWN:
        #if test == True:
                #Checking if coursor is above current button   
                if check_if_over(*pos_cursor, tile):
                    #Determine index of tile which is clicked
                    swapped_tile_idx = index
                    
                    #Check if clicked tile is next to the blank tile 
                    if check_next_blank(blank_tile_idx, swapped_tile_idx):
                        #Update puzzle (array not tiles)
                        new_puzzle = GameDynamics.swap_tiles(new_puzzle, blank_tile_idx, swapped_tile_idx)
                        
                        #Swap tiles and update blank tile index
                        tiles, blank_tile_idx = VisualizeGame.update_tiles(tiles, blank_tile_idx, swapped_tile_idx)
                        
                        #Check if game has been won 
                        won = GameDynamics.check_won(solution, new_puzzle)

        #Draw the tile in the window
        tile.draw(state_VG.win, (0,0,0))
    
    #Show updates
    pygame.display.update()


    return GameDynamicsState(new_puzzle, solution, blank_tile_idx, swapped_tile_idx, won), VisualizeGameState(tiles, win)




