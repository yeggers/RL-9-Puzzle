# Solving the 9-Puzzle with Reinforcement Learning

This reposotory contains the code to obtain a solution to all initial configurations of the 9-Puzzle. Furthermore, it also contains the code to play the game manually.

## Training the Puzzle 
In order to perform the training, run the file 'reinforcement_learning.py'. During the first run the initialization can take a few minutes. After running the code for the first time set 'LOAD_INITIALIZATION = True' and the initialization results will be read from a file. To perform the sensitivity analysis, specify which parameter is being varied by setting the variable 'parameter' to either 'E', 'GAMMA' or R_FINAL. After all test runs have been performed the results can be plotted by running 'sensitivity_analysis.py'. 

## Playing the Game 
In order to play the game manually, run 'play_manually.py' and the game window will pop up.
