
import numpy as np
import pygame

import game.game_dyanmics as game_dynamics
import game.play_manually as play_manually
import game.visualize_game as visualize_game

#Define parameters 
N = 3                                 #Number of rows/columns in puzzle [-]

TILE_SIZE = 150                       #Size of tile in puzzle [pixels]
TILE_COLOR = (0,255,0)                #Color of tiles
WINDOW_SIZE = N * TILE_SIZE           #Size of game window [pixles]

#Initialize outer game loop 
run = True
while run:
    #Initialize Game parameters 
    GameDynamicsParameters = game_dynamics.GameDynamicsParameters(N)
    #Initialize game 
    GameDynamics = game_dynamics.GameDyanmics(GameDynamicsParameters)
    #Initialize Game states 
    GameDynamicsStates = game_dynamics.GameDynamicsState(*GameDynamics.make_new_puzzle(), [])
   

    #Initialize visualization parameters
    VisualizeGameParameters = visualize_game.VisualizeGameParameters(TILE_COLOR, TILE_SIZE, WINDOW_SIZE)
    #Initialize visualization
    VisualizeGame = visualize_game.VisualizeGame(GameDynamicsParameters, VisualizeGameParameters)
    #Initialize visualization state
    VisualizeGameStates = visualize_game.VisualizeGameState(*VisualizeGame.draw_inital_puzzle(GameDynamicsStates))

    while GameDynamicsStates.won == False:
        for event in pygame.event.get():
            
            #Make sure game can be exited 
            if event.type == pygame.QUIT:
                run = False 
                pygame.quit()
                quit()
                
            #Determine which tile is clicked, swap and check if game has been won 
            GameDynamicsStates, VisualizeGameStates = play_manually.determine_swapped_tile_idx(GameDynamics, VisualizeGame, GameDynamicsStates, VisualizeGameStates, event)

    #Show winning screen if game has been won 
    new_game = False 
    while new_game == False:
        for event in pygame.event.get(): 
            new_game, run = VisualizeGame.draw_winning_screen(VisualizeGameStates, event, new_game, run)







