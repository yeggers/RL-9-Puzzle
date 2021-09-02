
import itertools
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame

import game.game_dyanmics as game_dyanmics
import game.play_manually as play_manually


class Tile():
    """ 
       Define properties of a tile in the gaming window

        Parameters: 
          color (tuple): desired colour of the tile
          x (int): x-position of the tile
          y (int): y-position of the tile
          tile_width: (int): width of the tile in pixels
          tile_height: (int): height of the tile in pixels
          text (str): text to be printed on the tiles
    
    """      
    def __init__(self, color, x, y, tile_width, tile_height, text = ''):
        self.color = color
        self.x = x
        self.y = y
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.text = text
        self.xrange = np.arange(self.x, self.x + self.tile_width)
        self.yrange = np.arange(self.y, self.y + self.tile_height)

    
    def draw(self, win, outline=None):
        """ 
        Draw the tile in the pygame window
            
            Parameters: 
              win (pygame.display.set_mode): pygame window
        
        """
       
        if outline:
            pygame.draw.rect(win, outline, (self.x-2, self.y-2, self.tile_width + 4, self.tile_height + 4),0)
            
        pygame.draw.rect(win, self.color, (self.x, self.y, self.tile_width, self.tile_height), 0)
        
        if self.text != '':
            font = pygame.font.SysFont('comicsans', 60)
            text = font.render(self.text, 1, (0,0,0))
            win.blit(text, (self.x + (self.tile_width/2 - text.get_width()/2), self.y + (self.tile_height/2 - text.get_height()/2)))
   


class VisualizeGameParameters(NamedTuple):
    """Parameters of the 15-puzzle.

    Parameters:
      color (tuple): desired colour of the tiles
      tile_size (int): desired size of tiles
      window_size (int): size of window for game 
    
    """

    color: tuple
    tile_size: int
    window_size: int
   
 

class VisualizeGameState(NamedTuple):
    """State of the 15-puzzle.

        Parameters:
          tiles (list): list containing instances of each tile in the game
          win (pygame.display.set_mode): game window
    
    """

    tiles: list 
    win: pygame.display.set_mode


class VisualizeGame():
    """ 
       Visualize the puzzle and move tiles acording to user input

        Parameters: 
          p_GD (GameDyanmicsParameters): game parameters
          p_VG (VisualizeGameParameters): visualization parameters
            
    """
    def __init__(self, p_GD, p_VG):
        self.p_GD = p_GD
        self.p_VG = p_VG
      
   
    def draw_inital_puzzle(self, state_GD):
        """
            Draw the initial puzzle

            Parameters: 
               state_GD (GameDynamicsState): state of the game

        """
        pygame.init()

        #Draw window of puzzle and fill it
        win = pygame.display.set_mode((self.p_VG.window_size, self.p_VG.window_size))
        win.fill((255, 255, 255))  

        #Create an array containing the positions of the buttons
        pos_tiles = itertools.product(np.arange(0, self.p_VG.window_size, self.p_VG.tile_size), repeat = 2)

        #Create an instance of the Tile class for each tile 
        tiles = []
        for pos_tile in pos_tiles:
            tiles.append(Tile(self.p_VG.color, pos_tile[1], pos_tile[0], self.p_VG.tile_size, self.p_VG.tile_size, str(state_GD.new_puzzle[int(pos_tile[0]/self.p_VG.tile_size), int(pos_tile[1]/self.p_VG.tile_size)])))
        
        #Reshape list of tiles to match layout of puzzle
        tiles = np.array(tiles).reshape(self.p_GD.N, self.p_GD.N)

        #Make blank tile black
        tiles[state_GD.blank_tile_idx].color = (0,0,0)

        return tiles, win


    def update_tiles(self, tiles, blank_tile_idx, swapped_tile_idx):
        """
            Update tiles based on user input 

            Parameters: 
                tiles (list): list of tile instances 
                blank_tile_idx (tuple): index of the blank tile
                swapped_tile_idx (tuple): index of the swapped tile

        """
    
        #Swap tiles
        blank_tile_text = tiles[blank_tile_idx].text
        blank_tile_color = tiles[blank_tile_idx].color
       
        tiles[blank_tile_idx].text = tiles[swapped_tile_idx].text
        tiles[blank_tile_idx].color = tiles[swapped_tile_idx].color
        tiles[swapped_tile_idx].text = blank_tile_text
        tiles[swapped_tile_idx].color = blank_tile_color

        #Determine new blank tile index
        blank_tile_idx = swapped_tile_idx
    
        return tiles, blank_tile_idx

    
    def draw_winning_screen(self, state_VG, event, new_game, run):
        """
            Draw screen after game has been wone and ask user whether or not to play another game

            Parameters: 
               state_VG (VisualizeState): state of the visualization
               event (pygame.event.get): pygame event
               new_game (bool): bool specifying whether new game should be initialized
               run (bool): bool specifying whether game should be running

        """
        #Creating white screen saying 'You won'.
        pygame.draw.rect(state_VG.win, [255, 255, 255], [0, 0, self.p_GD.N*self.p_VG.tile_size, self.p_GD.N*self.p_VG.tile_size], 0)
        font = pygame.font.SysFont('comicsans', 60)
        text = font.render('YOU WON!', 1, (255,0,0))
        state_VG.win.blit(text, (self.p_GD.N*self.p_VG.tile_size/2 - text.get_width()/2, self.p_GD.N*self.p_VG.tile_size/2 - text.get_height()/2-30))
        
        #Creating 'Again'- and 'Exit'-button.
        Again = Tile((0,255,0), self.p_GD.N*self.p_VG.tile_size/2 -190, self.p_GD.N*self.p_VG.tile_size/2 + 60, 150, 50, 'Again' )
        Exit = Tile((0,255,0), self.p_GD.N*self.p_VG.tile_size/2 + 40, self.p_GD.N*self.p_VG.tile_size/2 + 60, 150, 50, 'Exit' )
        Again.draw(state_VG.win, (0,0,0))
        Exit.draw(state_VG.win, (0,0,0))
        
        #Displaying changes
        pygame.display.update()

        #Check if mouse button is pressed.
        if event.type == pygame.MOUSEBUTTONDOWN:
            #Determine position of mouse button 
            pos_cursor = pygame.mouse.get_pos()
            
            #Check if coursor is over 'Exit'-button
            if play_manually.check_if_over(*pos_cursor, Exit):
                new_game = False
                run = False 
                #Quitting game
                pygame.quit()
                quit()
            
            #Check if coursor is over 'Again'-button    
            if play_manually.check_if_over(*pos_cursor, Again):
                #Returning to outer loop to create new game.
                new_game = True
                run = True
        
        return new_game, run

  




  





