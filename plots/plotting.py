import matplotlib.pyplot as plt 
import numpy as np 


def plot_puzzle_histogram(num_steps):
    """
        Plot how many puzzles can be solved within certain number of steps
        
        Parameters: 
                num_steps (np.array): array specifying how many steps are required to solve the puzzle for each state
    """

    #Remove non-solvable puzzles from the num_steps array 
    num_steps = num_steps[np.where(num_steps != np.max(num_steps))]

   

    #Plot histogram 
    plt.figure()
    plt.hist(num_steps, int(np.max(num_steps))+1)
    plt.xlabel('Number of steps [-]')
    plt.ylabel('Number of states [-]')
    plt.grid()
    #plt.show()

    
def plot_value_function(V, NUM_STATES):
    """
        Create scatter plot of value function values
        
        Parameters: 
                V (np.array): value function
    """

    #Set up x-array with total number of states 
    x = np.arange(NUM_STATES)

    labels = [str(i).zfill(2) for i in range(1, len(V)+1)]

    # #Normalize V
    # V = (V - np.min(V))/(np.max(V) - np.min(V))

    
    fig = plt.figure()
    ax = plt.gca()
    for index, value in enumerate(V):
        ax.scatter(x, value.flatten(), label = 'i =  ' + labels[index])
    #ax.set_yscale('log')
    ax.set_xlabel('State index (flattened) [-]')
    ax.set_ylabel('Value function [-]')
    #plt.legend()
    plt.grid()
#
    
    #plt.show()







if __name__ == "__main__":
    num_steps = np.load('results/num_steps/num_steps_N3_E1_GAMMA_0.9999_2021-08-29 13:27:24.844659.npy')
    plot_puzzle_histogram(num_steps)

    # NUM_STEPS = np.math.factorial(9)
    # V3 = np.load('results/value_functions/value_function_N3_E_15_GAMMA1_R_FINAL-10000_2021-08-29 04:54:56.762481.npy')
    # V2 = np.load('results/value_functions/value_function_N3_E_15.8_GAMMA1_R_FINAL0_2021-08-29 01:22:14.383227.npy')
    # V4 = np.load('results/value_functions/policy_N3_E_15_GAMMA1.npy')
    # plot_value_function(V2, NUM_STEPS)
    # plot_value_function(V3, NUM_STEPS)
    # plot_value_function(V4, NUM_STEPS)

    plt.show()