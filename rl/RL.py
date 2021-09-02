from pyvis.network import Network
import game.game_dyanmics as game_dynamics
from itertools import permutations
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy.core.numeric import indices

plt.rcParams.update({'font.size': 20})


class RLParameters(NamedTuple):
    """Reinforcement learning parameters.

    Parameters:
      N (int): number of rows/columns in the puzzle 
      MIN (int): minimum value for random initial value function 
      NUM_STATES (int): total number of states
      MAX (int): maximum value for random initial value function
      DIM (tuple): dimensions of the state 
      R_FINAL (int): reward for final state
      GAMMA (int): learning rate 

    """

    N: int
    DIM: tuple
    NUM_STATES: int
    V_MIN: int
    V_MAX: int
    R_FINAL: int
    GAMMA: float


class RLState(NamedTuple):
    """Reinforcement learning states.

    Parameters:
       V (np.array): value function 
    """

    V: np.array


class RL():
    """
    Perform reinforcement learning on 15-puzzle.

        Parameters: 
          p (RLParameters): parameters of the RL algorithm

    """

    def __init__(self, p):
        self.p = p

    def set_up_state_array(self):
        """
        Initialize value function

            Parameters: 

        """
        states = np.array(list(permutations(range(0, self.p.N**2)))
                          ).reshape(*self.p.DIM, self.p.N, self.p.N)

        return states

    def initialize_value_function(self):
        """
        Initialize value function

            Parameters: 

        """

        # Fill array of dimension dim with random integers
        V = np.random.randint(self.p.V_MIN, self.p.V_MAX, (self.p.DIM))
        #Initialize value function of final state with large negative number
        V[tuple(np.zeros(self.p.N**2).astype(int))] = self.p.R_FINAL

        return V

    def compute_reward(self, states):
        """
        Compute reward of each state. The reward is the sum of the x- and y-distances between the desired and current position of the tiles

        Parameters: 
            states (np.array): array containing all possible states of the game


    """

        # Create list of indices. The blank tile is assigned number N
        indices_2D = np.dstack(np.indices((self.p.N, self.p.N)))

        # Flatten to 2 dimensions so values in puzzle can be used as indices
        indices = indices_2D.reshape(self.p.N*self.p.N, 2)

        # #Compute reward
        # value_indices =  np.resize(indices, (N*N, N*N,2))
        # position_indices = np.transpose(value_indices, [1,0,2])

        # #Compute difference between desired and actual indices
        # reward_grid = np.sum(abs(value_indices - position_indices), axis = 2)
        # print(reward_grid)

        # Determine the desired position of each tile in every state
        desired_indices = indices[states]

        # Compute distance of desired state to actual state
        R = abs(desired_indices - indices_2D)

        # Sum x and y distances and all distances for each state
        R = np.sum(R, axis=tuple(
            np.arange(self.p.N**2, len(desired_indices.shape))))

        # #Assign negative reward to final state
        # R[tuple(np.zeros(self.p.N**2).astype(int))] = self.p.R_FINAL

        return R

    def compute_probabilities(self, states):
        """
        Compute probabilities to transition from state s to s'. The four states are left, right, up down. Transition is not possible if the blank tile (N**2 -1) is positioned in left/right column or uppest/lowest row respoectively. Then P = 0, otherwise P = 1.
        With state dimensions state.shape = (N**2, N**-1, ..., 1, N, N) the axes of the rows and columns correspond to axis_row = N**2 and axis_column = N**2 +1. The indices of the of the first and last row/column are i_first = 0 and i_last = N -1.


            Parameters: 
                states (np.array): array containing all possible states of the game 

        """
        # Set probability of staying in current position to zero for all states but the final one
        P_zero = np.zeros(self.p.DIM)
        P_zero[tuple(np.zeros(self.p.N**2).astype(int))] = 1

        # Initialize remainng state transitioning probabilities to ones for all states
        P_left = np.ones(self.p.DIM)
        P_left[tuple(np.zeros(self.p.N**2).astype(int))] = 0
        P_right = np.ones(self.p.DIM)
        P_right[tuple(np.zeros(self.p.N**2).astype(int))] = 0
        P_up = np.ones(self.p.DIM)
        P_up[tuple(np.zeros(self.p.N**2).astype(int))] = 0
        P_down = np.ones(self.p.DIM)
        P_down[tuple(np.zeros(self.p.N**2).astype(int))] = 0

        # Find indices of states where blank tile (N**2 -1) cannot be moved in direction of action
        indices_left = np.where(
            np.take(states, indices=0, axis=self.p.N**2 + 1) == self.p.N**2-1)
        indices_right = np.where(
            np.take(states, indices=self.p.N-1, axis=self.p.N**2 + 1) == self.p.N**2-1)
        indices_up = np.where(
            np.take(states, indices=0, axis=self.p.N**2) == self.p.N**2-1)
        indices_down = np.where(
            np.take(states, indices=self.p.N-1, axis=self.p.N**2) == self.p.N**2-1)

        # Setting probabilities of states for which blank tile cannot be moved in direction of action to zero.Last two indices of indices are not required since they represent the position of the numbers in the grid for each state. (The first index was already lost with np.take )
        P_left[indices_left[:-1]] = 0
        P_right[indices_right[:-1]] = 0
        P_up[indices_up[:-1]] = 0
        P_down[indices_down[:-1]] = 0

        return (P_zero, P_left, P_right, P_up, P_down)

    def initialize_policy(self, P_a):
        """
        Initialize policies for the 4 actions. Policies are 
            - zero: stay in current position 
            - left: move blank tile to the left 
            - right: move blank tile to the right 
            - up: move blank tile up 
            - down: move blank tile down

            Parameters: 
                P_a (tuple): Tuple containing the state transitioning probabilities (np.array) for all actions

        """

        # Determine how many action options there are for every state
        number_of_options = sum(P_a)

        # Action probabilities corresponding to random action
        P_a_rand = 1/number_of_options

        # #Setting action probabilities for final state to zero
        # P_a_rand[tuple(np.zeros(self.p.N**2).astype(int))] = 0

        # Initialize policy to stay in current position to zero for all states but the final one
        K_zero = P_a_rand.copy()
        K_left = P_a_rand.copy()
        K_right = P_a_rand.copy()
        K_up = P_a_rand.copy()
        K_down = P_a_rand.copy()

        return (K_zero, K_left, K_right, K_up, K_down)

    def compute_shifted_indices(self, states, P_a):
        """
            Compute indices for shifts

            Parameters: 
                states (np.array): array containing all possible states of the game
                P_a (np.array): state transitioning probabilities for left action 


        """

        print(' ')
        print('Compute shifted indices. This can take a few minutes but only needs to be done once. After code has been run for first time set LOAD_INITIALIZATION = True.')

        # Determine indices for which tiles can be swapped
        indices_left = np.where(P_a[1])
        indices_right = np.where(P_a[2])
        indices_up = np.where(P_a[3])
        indices_down = np.where(P_a[4])

        # Compute indices of blank tiles for each state
        blank_indices_left = np.where(states[indices_left] == self.p.N**2-1)
        blank_indices_right = np.where(states[indices_right] == self.p.N**2-1)
        blank_indices_up = np.where(states[indices_up] == self.p.N**2-1)
        blank_indices_down = np.where(states[indices_down] == self.p.N**2-1)

        # Initializing shifted states as original states
        states_left = states.copy()
        states_right = states.copy()
        states_up = states.copy()
        states_down = states.copy()

        # Applying shifts where possible
        states_left[(*indices_left, *blank_indices_left[1:])] = states_left[(*
                                                                             indices_left, blank_indices_left[1], blank_indices_left[2] - 1)]
        states_left[(*indices_left, blank_indices_left[1],
                     blank_indices_left[2]-1)] = self.p.N**2 - 1

        states_right[(*indices_right, *blank_indices_right[1:])] = states_right[(
            *indices_right, blank_indices_right[1], blank_indices_right[2] + 1)]
        states_right[(*indices_right, blank_indices_right[1],
                      blank_indices_right[2]+1)] = self.p.N**2 - 1

        states_up[(*indices_up, *blank_indices_up[1:])] = states_up[(*
                                                                     indices_up, blank_indices_up[1] - 1, blank_indices_up[2])]
        states_up[(*indices_up, blank_indices_up[1] - 1,
                   blank_indices_up[2])] = self.p.N**2 - 1

        states_down[(*indices_down, *blank_indices_down[1:])] = states_down[(*
                                                                             indices_down, blank_indices_down[1] + 1, blank_indices_down[2])]
        states_down[(*indices_down, blank_indices_down[1] + 1,
                     blank_indices_down[2])] = self.p.N**2 - 1

        number_of_states = np.math.factorial(self.p.N**2)

        shifted_indices_left = np.zeros((self.p.N**2, number_of_states))
        shifted_indices_right = np.zeros((self.p.N**2, number_of_states))
        shifted_indices_up = np.zeros((self.p.N**2, number_of_states))
        shifted_indices_down = np.zeros((self.p.N**2, number_of_states))

        for i in range(number_of_states):
            idx_left = self.determine_index(
                states_left.reshape(number_of_states, self.p.N**2)[i])
            idx_right = self.determine_index(
                states_right.reshape(number_of_states, self.p.N**2)[i])
            idx_up = self.determine_index(
                states_up.reshape(number_of_states, self.p.N**2)[i])
            idx_down = self.determine_index(
                states_down.reshape(number_of_states, self.p.N**2)[i])

            shifted_indices_left[:, i] = idx_left
            shifted_indices_right[:, i] = idx_right
            shifted_indices_up[:, i] = idx_up
            shifted_indices_down[:, i] = idx_down

           

        np.save('results/shifted_indices/shifted_indices_left_N{N}.npy'.format(
            N=self.p.N), shifted_indices_left)
        np.save('results/shifted_indices/shifted_indices_right_N{N}.npy'.format(
            N=self.p.N), shifted_indices_right)
        np.save(
            'results/shifted_indices/shifted_indices_up_N{N}.npy'.format(N=self.p.N), shifted_indices_up)
        np.save('results/shifted_indices/shifted_indices_down_N{N}.npy'.format(
            N=self.p.N), shifted_indices_down)

        np.save(
            'results/shifted_states/shifted_states_left_N{N}.npy'.format(N=self.p.N), states_left)
        np.save(
            'results/shifted_states/shifted_states_right_N{N}.npy'.format(N=self.p.N), states_right)
        np.save(
            'results/shifted_states/shifted_states_up_N{N}.npy'.format(N=self.p.N), states_up)
        np.save(
            'results/shifted_states/shifted_states_down_N{N}.npy'.format(N=self.p.N), states_down)

        return (tuple(shifted_indices_left.astype(int)), tuple(shifted_indices_right.astype(int)), tuple(shifted_indices_up.astype(int)), tuple(shifted_indices_down.astype(int))), (states_left, states_right, states_up, states_down)

    def determine_index(self, state):
        """
        Determine index of a certain state

            Parameters: 
                state (np.array): state whose index is to be determined


        """
        indices = []
        numbers = np.arange(self.p.N**2)

        for i in range(self.p.N**2):
            index = np.where(numbers == state[i])
            indices.append(index)
            numbers = np.delete(numbers, index)

        return np.array(indices).squeeze()

    def update_V(self, K_a, P_a, R_a, V_a):
        """
        Update value function

            Parameters: 
                K_a (tuple): policies for all actions
                P_a (tuple): state transitioning probabilities for all actions 
                R_a (tuple): shifted rewards for all actions
                V_a (tuple): shifted value functions for all actions

        """

        V_new = (K_a[0] * (P_a[0] * (R_a[0] + self.p.GAMMA*V_a[0]))
                 + K_a[1] * (P_a[1] * (R_a[1] + self.p.GAMMA*V_a[1]))
                 + K_a[2] * (P_a[2] * (R_a[2] + self.p.GAMMA*V_a[2]))
                 + K_a[3] * (P_a[3] * (R_a[3] + self.p.GAMMA*V_a[3]))
                 + K_a[4] * (P_a[4] * (R_a[4] + self.p.GAMMA*V_a[4]))).reshape(self.p.DIM)

       
        V_new[tuple(np.zeros(self.p.N**2).astype(int))] = self.p.R_FINAL

        return V_new

    def update_policy(self, s, shifted_indices):
        """
        Update policy based on new value function 

            Parameters: 
            s (RLState): state of the reinforcement learning approach (V)

        """

        # Determine shifted value function
        V_left = s.V[shifted_indices[0]].reshape(self.p.DIM)
        V_right = s.V[shifted_indices[1]].reshape(self.p.DIM)
        V_up = s.V[shifted_indices[2]].reshape(self.p.DIM)
        V_down = s.V[shifted_indices[3]].reshape(self.p.DIM)

        # Determine value of lowest unsolvable state
        lowest_unsolvable = np.sort(s.V, axis=None)[int(self.p.NUM_STATES/2)]
        test = np.where(s.V == lowest_unsolvable)

        # Determine indices of non-solvable states
        unsolvable_indices = np.where(s.V >= lowest_unsolvable)

        # Set up arary to collect results
        R_greedy = np.zeros_like(s.V)

        # Check which value function is the lowest in direct neighbourhood of each state and assign values
        # to each action:
        # 1: left
        # 2: right
        # 3: up
        # 4: down
        # 5: unsolvable

        R_greedy[np.where(np.logical_and(np.logical_and(np.less(V_left, s.V), np.less_equal(
            V_left, V_right)), np.logical_and(np.less_equal(V_left, V_up), np.less_equal(V_left, V_down))))] = 1
        R_greedy[np.where(np.logical_and(np.logical_and(np.less(V_right, s.V), np.less_equal(
            V_right, V_left)), np.logical_and(np.less_equal(V_right, V_up), np.less_equal(V_right, V_down))))] = 2
        R_greedy[np.where(np.logical_and(np.logical_and(np.less(V_up, s.V), np.less_equal(
            V_up, V_right)), np.logical_and(np.less_equal(V_up, V_left), np.less_equal(V_up, V_down))))] = 3
        R_greedy[np.where(np.logical_and(np.logical_and(np.less(V_down, s.V), np.less_equal(
            V_down, V_right)), np.logical_and(np.less_equal(V_down, V_up), np.less_equal(V_down, V_left))))] = 4
        #R_greedy[unsolvable_indices] = 5

        return unsolvable_indices, R_greedy

    def count_steps_to_solution(self, policy, shifted_indices, N_max):
        """
        Count how many steps are required for each state to reach the solution 

            Parameters: 
                policy (np.array): policy for each state
                shifted_indices: (np.array): array describing indices of shifted states  
                N_max (int): maximum number of steps   
        """

        # Create array containing index of each state
        new_indices = np.indices(self.p.DIM).reshape(
            self.p.N**2, self.p.NUM_STATES)

        # Create array containing indices of all states + shifted states with shape
        all_indices = np.array([new_indices, *shifted_indices]
                               ).transpose((0, 2, 1)).reshape(5, *self.p.DIM, self.p.N**2)

        new_indices = tuple(map(tuple, new_indices))

        # Set up empty array to collect policy steps
        policy_steps = []

        # Set up counter for number of required steps
        counter = np.zeros(self.p.NUM_STATES)

        # Check with states have not reached final state
        check = (np.array(new_indices) == 0).all(axis=0)

        test = 0
        # while np.sum(check)< self.p.NUM_STATES/2:
        while test < N_max:
            test += 1
            print('number of steps: ', test)
            print('puzzles solved', np.sum(check))
            print(' ')
            # Append policy of transitioned state
            policy_steps.append(policy[new_indices].reshape(self.p.DIM))

            # Check which states have reached final state
            check = (np.array(new_indices) == 0).all(axis=0)

            # Compute indices to determine new indices
            i = (tuple(policy[new_indices].astype(int)), *new_indices)

            # Compute new indices
            new_indices = tuple(map(tuple, all_indices[i].transpose(1, 0)))

            # Increment counter of states that have not reached final state
            counter += (~check).astype(int)

        # Determine number unsolved puzzles
        num_unsolved = self.p.NUM_STATES/2 - np.sum(check)

        print('puzzles not solved: ', num_unsolved)

        return np.array(policy_steps).transpose(*tuple(np.arange(1, self.p.N**2+1)), 0), counter.reshape(self.p.DIM), num_unsolved

    def determine_deadends(self, policy, solvable):
        """
        Determine if/how many deadends the policy includes. Non-solvable states are disregarded

            Parameters: 
                policy (np.array): policy for each state
                solvable: (np.array): array specifying for each state whether it is solvable    
        """

        test = np.where(solvable)
        num_deadends = len(np.where(policy[np.where(solvable)] == 0)[0])

        return num_deadends

    def determine_solvable_states(self):
        """
        Compute which states are solvable

            Parameters: 

        """
        print(' ')
        print('Determining indices of unsolvable states manually. This can take a few minutes but only needs to be done once. After code has been run for first time set LOAD_INITIALIZATION = True.')

        # Set up array containing all permutations and add 1 to make it compatible with function that checks for solvability
        states = np.array(list(permutations(range(0, self.p.N**2)))) + 1

        # Use zero rather than N**2 as the blank tile
        states[np.where(states == self.p.N**2)] = 0

        # Initialize array to collect results
        solvable = np.zeros(self.p.NUM_STATES)

        # Initialize GameDynamics class which is used to compute solvability
        GameDynamics = game_dynamics.GameDyanmics(
            game_dynamics.GameDynamicsParameters(self.p.N))

        for index, value in enumerate(states):
            solvable[index] = GameDynamics.check_if_solvable_b(value)[0]

        np.save('results/solvable_N{N}'.format(N=self.p.N), solvable)

        return solvable

    def determine_closest_states(self, N, num_steps):
        """
        Determine the N closest states to the solution state.

        Parameters: 
                N (int): number of closest states
                num_steps (np.array): array specifing how many steps are required to arrive at the solution for every state
        """

        # Determine how often each step count occurs
        unique, count = np.unique(num_steps, return_counts=True)
        # Compute how many states can be solved in fewer or same number of steps for every step
        num_step_int = np.cumsum(count)
        # Compute what is the maximum number of steps to arrive at less than 100 states
        step_max = np.max(np.where(num_step_int < N))
        # Determine indices of 100 closest states
        indices_N = np.where(num_steps <= step_max)

        return indices_N

    def plot_unsolvable_indices(self, indices_RL, indices_HC):
        """
        Plot the indices of the unsolvable states determined with the RL algorithm and by bruteforce.

        Parameters: 
                indices_RL (tuple): indices as determined by the RL-algorithm
                indices_HC (np.array): array specifying which states are solvable
        """
        # Determine indices of hardcoded input
        indices_HC = np.where(indices_HC == False)
        xlimits = np.array(
            [[181450, 181450, 30000], [3000, 700, 200], [50, 25, 50]])

        for i in range(self.p.N):
            for j in range(self.p.N):
                plt.subplot(self.p.N, self.p.N, i*self.p.N + j + 1)
                plt.plot(indices_HC[i*self.p.N + j], color='k',
                         label='Manually computed', linewidth=4)
                plt.plot(indices_RL[i*self.p.N + j], color='lime',
                         linestyle='--', label='Computed with RL', linewidth=3.2)
                plt.xlabel('State number [-] ')
                plt.ylabel(
                    'Index {dim}. dim. [-]'.format(dim=i*self.p.N + j+1))
                plt.yticks(tuple(np.arange(self.p.N**2)))
                plt.xlim((0, xlimits[i, j]))
                plt.grid()
        plt.legend()

    def draw_connections(self, states, shifted_states, policy, s, NUM_STATES, indices=None):

        # Shift states according to new policy
        all_states = np.array([states, *shifted_states])
        all_indices = tuple(map(tuple, np.where(np.ones(self.p.DIM))))

        if indices == None:
            shifted_states = all_states[(
                tuple(policy.flatten()), *all_indices)]
        else:
            shifted_states = all_states[(
                tuple(policy[indices].flatten()), *indices)]

        # Convert states and shifted states to nicer format
        states = states + 1
        states[np.where(states == self.p.N**2)] = 0

        shifted_states = shifted_states + 1
        shifted_states[np.where(shifted_states == self.p.N**2)] = 0

        # Set up dictionary describing mapping between states and shifted states (converting to strings since Networks can't handle nump arrays)
        connections = {
            'States': [np.array2string(row) for row in states[indices].reshape(NUM_STATES, self.p.N, self.p.N)],
            'Shifted States': [np.array2string(row) for row in shifted_states]
        }

        # Convert to data drame
        connections_data_frame = pd.DataFrame(
            connections, columns=['States', 'Shifted States'])

        # Set up network
        G = nx.from_pandas_edgelist(
            connections_data_frame, source='States', target='Shifted States', create_using=nx.DiGraph())

        # Determine positions of nodes
        pos = nx.spring_layout(G)

        # Moving position of text
        pos_higher = {}
        y_off = 0.02
        x_off = 0.02

        for k, v in pos.items():
            pos_higher[k] = (v[0] + x_off, v[1]+y_off)
        # Create text labels displaying the value function for each state
        V_strings = [str(value)
                     for value in np.round(s.V[indices].flatten(), 2)]

        # Create dictionary
        labels = dict(zip([np.array2string(row) for row in states[indices].reshape(
            NUM_STATES, self.p.N, self.p.N)], V_strings))

        # plt.figure()
        # # Draw network
        # nx.draw_networkx(G, pos_higher, node_size=1000, font_color='w')
        # nx.draw_networkx_labels(G, pos_higher, labels)

        net = Network(notebook=True)
        net.from_nx(G)
        net.show('results/connections.html')

       

    # Only for 4D-case
    def draw_initial_connections(self, states, shifted_states, R):

        states_reshaped = states.reshape(self.p.NUM_STATES, self.p.N, self.p.N)
        states_all = np.array([states_reshaped, states_reshaped, states_reshaped, states_reshaped]).reshape(
            4*self.p.NUM_STATES, self.p.N, self.p.N)
        shifted_states = np.array([shifted_states[0], shifted_states[1], shifted_states[2], shifted_states[3]]).reshape(
            4*self.p.NUM_STATES, self.p.N, self.p.N)

        connections = {
            'States': [np.array2string(row) for row in states_all],
            'Shifted States': [np.array2string(row) for row in shifted_states]
        }

        connections_data_frame = pd.DataFrame(
            connections, columns=['States', 'Shifted States'])

        #x1, y1, x2, y2 = make_circle_coordinates((0,0), (0, 300), 50, int(self.p.N**2/2))

        G = nx.from_pandas_edgelist(connections_data_frame, source='States', target='Shifted States',
                                    create_using=nx.DiGraph())

        # net = Network(notebook = True)
        # net.from_nx(G)
        # net.show('example.html')

        # nx.draw_networkx_edges(
        # G, pos,
        # edgelist=elist,
        # edge_color='green',
        # width=3,
        # label="S",
        # arrowstyle='-|>')
        plt.figure()
        # Determine positions of nodes
        pos = nx.spring_layout(G)

        nx.draw_networkx(G, pos, node_size=1500, font_color='w')

        # Moving position of text
        pos_higher = {}
        y_off = 0.05
        x_off = 0.05

        for k, v in pos.items():
            pos_higher[k] = (v[0] + x_off, v[1]+y_off)

        # Create text labels displaying the value function for each state
        R_strings = [str(value) for value in np.round(R.flatten(), 2)]

        # Create dictionary
        labels = dict(zip([np.array2string(row) for row in states.reshape(
            self.p.NUM_STATES, self.p.N, self.p.N)], R_strings))

        nx.draw_networkx_labels(G, pos_higher, labels)
        plt.show()
