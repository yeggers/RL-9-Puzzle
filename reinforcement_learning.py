from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import plots.plotting as plot
import rl.RL as rl

plt.rcParams.update({'font.size': 20})


#Specify parameters

N = 3                                   #Number of rows/columns in the game
DIM = tuple(np.arange(N**2, 0, -1))     #Dimensions of the array containing all possible states
NUM_STATES = np.math.factorial(N**2)    #Total number of states

V_MIN = 0                               #Minmum intial value in value function
V_MAX = 100                             #Maximum intial value in value function

R_FINAL = -10000                        #Value function at final state

GAMMA = 0.9999                          #Discount factor

E = np.inf                              #Convergence error

LOAD_TRAINED_FILES = False              #Start training using results of previous training
LOAD_INITIALIZATION = False             #Load some of the initialization results which take long to compute from files


parameter = 'test'                      #Parameter to be analyzed during sensitivity analysis ('E'/ 'GAMMA'/ 'R_FINAL'/'test')




# #Specify files containing results from previous training
# V = np.load('results/value_functions/value_function_N3_E_15_GAMMA1_R_FINAL-10000_2021-08-29 04:54:56.762481.npy')

# R_greedy = np.load('results/policies/policy_N3_E_15_GAMMA1_R_FINAL-10000_2021-08-29 04:54:56.764294.npy')

# K_a = (np.zeros(DIM),np.zeros(DIM), np.zeros(DIM),np.zeros(DIM),np.zeros(DIM))

# K_a[0][np.where(R_greedy == 0)] = 1
# K_a[1][np.where(R_greedy == 1)] = 1
# K_a[2][np.where(R_greedy == 2)] = 1
# K_a[3][np.where(R_greedy == 3)] = 1
# K_a[4][np.where(R_greedy == 4)] = 1


#Specify files containing initialization results which take long to compute.  
shifted_indices_left = tuple(np.load('results/shifted_indices/shifted_indices_left_N{N}.npy'.format(N=N)).astype(int))
shifted_indices_right = tuple(np.load('results/shifted_indices/shifted_indices_right_N{N}.npy'.format(N=N)).astype(int))
shifted_indices_up = tuple(np.load('results/shifted_indices/shifted_indices_up_N{N}.npy'.format(N=N)).astype(int))
shifted_indices_down = tuple(np.load('results/shifted_indices/shifted_indices_down_N{N}.npy'.format(N=N)).astype(int))
shifted_indices = (shifted_indices_left, shifted_indices_right, shifted_indices_up, shifted_indices_down)

shifted_states_left = np.load( 'results/shifted_states/shifted_states_left_N{N}.npy'.format(N = N))
shifted_states_right = np.load( 'results/shifted_states/shifted_states_right_N{N}.npy'.format(N = N))
shifted_states_up = np.load( 'results/shifted_states/shifted_states_up_N{N}.npy'.format(N = N))
shifted_states_down = np.load( 'results/shifted_states/shifted_states_down_N{N}.npy'.format(N = N))
shifted_states = (shifted_states_left, shifted_states_right, shifted_states_up, shifted_states_down)

solvable_states = np.load('results/solvable_N{N}.npy'.format(N = N)).reshape(DIM)


#Intialize parameters for RL
RLParameters = rl.RLParameters(N, DIM, NUM_STATES, V_MIN, V_MAX, R_FINAL, GAMMA)

#Initialize RL class
RL = rl.RL(RLParameters)

#Set up array containing all possible states
states = RL.set_up_state_array()

#Compute state transitioning probabilities 
P_a = RL.compute_probabilities(states)

if LOAD_TRAINED_FILES == False:
    #Initialize policy with equal probabilities for each action
    K_a = RL.initialize_policy(P_a)

    #Initialize RL state (value function)
    V = RL.initialize_value_function()
RLState = rl.RLState(V)

if LOAD_INITIALIZATION == False:
    #Determine solvable states (takes a few minutes) 
    solvable_states = RL.determine_solvable_states().reshape(DIM)
    
    #Determine indices of states shifted by the 4 respective actions (left, right, up, down)(takes a few minutes)
    shifted_indices, shifted_states = RL.compute_shifted_indices(states, P_a)

#Compute all possible states and corresponding rewards
R = RL.compute_reward(states)

#Compute shifted rewards 
R_zero = R
R_left = R[shifted_indices[0]].reshape(DIM)
R_right = R[shifted_indices[1]].reshape(DIM)
R_up = R[shifted_indices[2]].reshape(DIM)
R_down = R[shifted_indices[3]].reshape(DIM)

#Initialize counters
inner = 0
num_deadends = np.inf


#Initialize array to collect results
inner_log = []
error_solvable_high_log = []
error_unsolvable_mean_log = []
num_deadends_log = []
num_steps_log = []
num_unsolved_log = []
Delta_log = []
progress_log = []
V_log = [V]



#Perform outer loop until only final state is a deadend. (Only for parameters which leade to convergence)
while num_deadends > 1:
#Use this outer loop requirement for non-converging parameters
#for i in range(40):
    inner = 0
    
    #Initiali3e number of non-converged states
    Delta = np.inf
    
    #Perform inner loop until all solvable states (1/2 of all states) have reached the error E
    while Delta > NUM_STATES/2 :
        inner+=1
        
        #Determine shifted value function 
        V_zero = RLState.V
        V_left = RLState.V[shifted_indices[0]].reshape(DIM)
        V_right = RLState.V[shifted_indices[1]].reshape(DIM)
        V_up = RLState.V[shifted_indices[2]].reshape(DIM)
        V_down = RLState.V[shifted_indices[3]].reshape(DIM)

        #Update the value function
        #V_new = RL.update_V(K_a, P_a, (R_zero, R_left, R_right, R_up, R_down), (V_zero, V_left, V_right, V_up, V_down))
        V_new = RL.update_V(K_a, P_a, (R_zero, R_zero, R_zero, R_zero, R_zero), (V_zero, V_left, V_right, V_up, V_down))
        
        #Determine error
        error = abs(RLState.V - V_new)
        #Sort errors by magnitude
        sorted_error = np.sort(error, axis = None)
        #Compute mean of errors of solvable states
        error_solvable_mean = np.mean(sorted_error[:int(NUM_STATES/2)])
        error_solvable_high = sorted_error[int(NUM_STATES/2)-1]
        error_solvable_high_log.append(error_solvable_high)
        #Compute mean of errors of unsolvable states
        error_unsolvable_mean = np.mean(sorted_error[int(NUM_STATES/2):])
        error_unsolvable_mean_log.append(error_unsolvable_mean)
        #Determine the highest error within the solvable states
        highest_error = sorted_error[int(NUM_STATES/2-1)]
        
        #Compute how many of the solvable states have reached maximum error
        Delta = np.count_nonzero(abs(RLState.V - V_new)>E)
        Delta_log.append(Delta)

        # #Compute total error
        # progress = np.sum(abs(RLState.V - V_new))
        # progress_log.append(progress)

        RLState = rl.RLState(V_new)
        
        print('highest solvable', highest_error)
        print('error solvable mean', error_solvable_mean)
        print('delta: ',Delta)
        print('Number of deadends: ', num_deadends)
        print(' ')
    
        # #Save results every 1000 steps
        # if inner%1000 == 0:
        #     np.save('results/sensitivity_analysis/{parameter}/policy_N{N}_E_{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()), RLState.V)

    #Save how many iterations were required in inner loop
    inner_log.append(inner)

    #Save updated value function 
    V_log.append(RLState.V)
    
    #Determine greedy policy
    unsolvable_indices, R_greedy = RL.update_policy(RLState, shifted_indices)
    
    #Update the policy 
    K_a = (np.zeros(DIM),np.zeros(DIM), np.zeros(DIM),np.zeros(DIM),np.zeros(DIM))

    K_a[0][np.where(R_greedy == 0)] = 1
    K_a[1][np.where(R_greedy == 1)] = 1
    K_a[2][np.where(R_greedy == 2)] = 1
    K_a[3][np.where(R_greedy == 3)] = 1
    K_a[4][np.where(R_greedy == 4)] = 1

    #Determine number of deadends in puzzle 
    num_deadends = RL.determine_deadends(R_greedy, solvable_states)
    num_deadends_log.append(num_deadends)

    # #Determine how many puzzles can be solved within 50 steps
    # policy, num_steps, num_unsolved = RL.count_steps_to_solution(R_greedy, shifted_indices, 50)
    # num_steps_log.append(num_steps[np.where(num_steps != np.max(num_steps))]) #Only append solvable oens
    # num_unsolved_log.append(num_unsolved)

#Determine how many puzzles can be solved within 50 steps
policy, num_steps, num_unsolved = RL.count_steps_to_solution(R_greedy, shifted_indices, 50)
num_steps_log.append(num_steps)
num_unsolved_log.append(num_unsolved)

#Determine 100 states closes to final state
indices_100 = RL.determine_closest_states(100, num_steps)

#Draw connections of 100 closest states
RL.draw_connections(states, shifted_states, R_greedy.astype(int), RLState, len(indices_100[0]), indices = indices_100)


#Draw connections before learning (only for N = 2)
#RL.draw_initial_connections(states, shifted_states, R)


#Plot the distribution showing how many states require how many steps to solve
plot.plot_puzzle_histogram(num_steps)
plt.savefig('results/sensitivity_analysis/{parameter}/histogram_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.png'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))

# #Plot the distribution of the value function 
# plot.plot_value_function(V_log, NUM_STATES)
# plt.savefig('results/sensitivity_analysis/{parameter}/value_function_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.png'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))

# #Plot indices of unsolvable puzzles
# figure = plt.gcf()
# figure.set_size_inches(20, 17)
# plt.savefig('results/unsolvable_state_indices_N{N}.png'.format(N = N))
# RL.plot_unsolvable_indices(unsolvable_indices, solvable_states)


#Plot the distribution of the value function 
plot.plot_value_function([RLState.V], NUM_STATES)
figure = plt.gcf()
figure.set_size_inches(13, 10)
plt.savefig('results/sensitivity_analysis/{parameter}/value_function_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.png'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))

#Plot number of steps in inner loop
plt.figure()
plt.bar(np.arange(len(inner_log)), inner_log)
plt.title('Number of steps in inner loop')
plt.xlabel('Inner loop number [-]')
plt.ylabel('Number of steps [-]')
plt.grid()
plt.savefig('results/sensitivity_analysis/{parameter}/inner_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.pdf'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))

# #Plot mean of error of solvable states 
# plt.figure()
# plt.plot(error_solvable_high_log)
# plt.title('Error solvable ')
# plt.xlabel('Step [-]')
# plt.ylabel('Error [-]')
# plt.grid()
# plt.savefig('results/sensitivity_analysis/{parameter}/error_solvable_high_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.pdf'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))

# #Plot mean of error of unsolvable states
# plt.figure()
# plt.plot(error_unsolvable_mean_log)
# plt.title('Error unsolvable mean')
# plt.xlabel('Step [-]')
# plt.ylabel('Error [-]')
# plt.grid()
# plt.savefig('results/sensitivity_analysis/{parameter}/error_unsolvable_mean_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.pdf'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))

# #Plot number of deadends
# plt.figure()
# plt.bar(np.arange(len(num_deadends_log)), num_deadends_log)
# plt.title('Number of deadends')
# plt.xlabel('Outer loop Step [-]')
# plt.ylabel('Number of deadends [-]')
# plt.grid()
# plt.savefig('results/sensitivity_analysis/{parameter}/num_deadends_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.pdf'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))

# #Plot number of unsolved puzzles
# plt.figure()
# plt.bar(np.arange(len(num_unsolved_log)), num_unsolved_log)
# plt.title('Number of unsolved puzzles')
# plt.xlabel('Outer loop Step [-]')
# plt.ylabel('Number of unsolved puzzles [-]')
# plt.grid()
# plt.savefig('results/sensitivity_analysis/{parameter}/num_unsolved_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.pdf'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))


# #Plot Delta 
# plt.figure()
# plt.plot(Delta_log)
# plt.title('Delta')
# plt.xlabel('Step [-]')
# plt.ylabel('Delta [-]')
# plt.grid()
# plt.savefig('results/sensitivity_analysis/{parameter}/Delta_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.pdf'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))

# #Plot progress
# #Plot mean of error of unsolvable states
# plt.figure()
# plt.plot(progress_log)
# plt.title('Progress')
# plt.xlabel('Step [-]')
# plt.ylabel('Progress [-]')
# plt.grid()
# plt.savefig('results/sensitivity_analysis/{parameter}/progress_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.pdf'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()))


    
#Save value function 
np.save('results/sensitivity_analysis/{parameter}/value_function_N{N}_E_{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.npy'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()), RLState.V)

#Save policy 
np.save('results/sensitivity_analysis/{parameter}/policy_N{N}_E_{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}_{date}.npy'.format(parameter = parameter, N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL, date = datetime.now()), R_greedy)

#Save logs
np.save('results/sensitivity_analysis/{parameter}/inner_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter,N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL), np.array(inner_log))
np.save('results/sensitivity_analysis/{parameter}/error_solvable_high_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter,N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL), np.array(error_solvable_high_log))
np.save('results/sensitivity_analysis/{parameter}/error_unsolvable_mean_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter,N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL), np.array(error_unsolvable_mean_log))
np.save('results/sensitivity_analysis/{parameter}/num_deadends_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter,N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL), np.array(num_deadends_log))
np.save('results/sensitivity_analysis/{parameter}/num_steps_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter,N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL), np.array(num_steps_log))
np.save('results/sensitivity_analysis/{parameter}/num_unsolved_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter,N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL), np.array(num_unsolved_log))
np.save('results/sensitivity_analysis/{parameter}/Delta_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter,N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL), np.array(Delta_log))
np.save('results/sensitivity_analysis/{parameter}/progress_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter,N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL), np.array(progress_log))
np.save('results/sensitivity_analysis/{parameter}/V_log_N{N}_E{E}_GAMMA{GAMMA}_R_FINAL{R_FINAL}.npy'.format(parameter = parameter,N = N, E = E, GAMMA = GAMMA, R_FINAL = R_FINAL), np.array(V_log))



#Assign unsolvable states to a seperate group
R_greedy[np.where(solvable_states==False)] = 5 
#Plot policies of all states
plt.figure()
plt.scatter(np.arange(NUM_STATES), R_greedy.flatten())
plt.xlabel('State number (flattened)[-]')
plt.yticks((0,1,2,3,4,5), ('Stay', 'Move right', 'Move left', 'Move up', 'Move down', 'Unsolvable'), fontsize = 30)
plt.grid()
figure = plt.gcf()
figure.set_size_inches(21, 12)
plt.savefig('results/policy_N{N}.png'.format(N = N))


plt.show()



