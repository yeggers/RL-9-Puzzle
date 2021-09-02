import numpy as np 
import matplotlib.pyplot as plt 
import plots.plotting as plot
plt.rcParams.update({'font.size': 20})




def plot_sensitivty_analysis(value_range, parameter):
    """
        Plot results of sensitivity analysis for one state.

        Parameters: 
                value_range (list): list of values which were investigated
                parameter (str): parameter which was investigated
        """

    #Set up arrays to collect results 
    error_logs = []
    inner_logs = []
    num_deadends_logs = []
    num_steps_logs = []
    num_unsolved_logs = []
    V_logs = []

    order_E = np.array([4,3,2, 1, 0])
    width_E = np.array([0.55, 0.575, 0.65, 0.725, 0.8])

    width_GAMMA = [0.725, 0.65, 0.6, 0.3, 0.8]
    order_GAMMA = [1,2,3,4,0]

    width_R_FINAL = [0.8, 0.725, 0.65, 0.6, 0.5]
    order_R_FINAL = [0,1,2,3,4]

    for index, value in enumerate(value_range):
        error_log = np.load('results/sensitivity_analysis/{parameter}/{parameter}{value}/error_solvable_high_log_N3_Einf_GAMMA{value}_R_FINAL-10000.npy'.format(value = value, parameter = parameter))
        error_logs.append(error_log)
        inner_log = np.load('results/sensitivity_analysis/{parameter}/{parameter}{value}/inner_log_N3_Einf_GAMMA{value}_R_FINAL-10000.npy'.format(value = value, parameter = parameter))
        inner_logs.append(inner_log)
        num_deadends_log = np.load('results/sensitivity_analysis/{parameter}/{parameter}{value}/num_deadends_log_N3_Einf_GAMMA{value}_R_FINAL-10000.npy'.format(value = value, parameter = parameter))
        num_deadends_logs.append(num_deadends_log)
        num_steps_log = np.load('results/sensitivity_analysis/{parameter}/{parameter}{value}/num_steps_log_N3_Einf_GAMMA{value}_R_FINAL-10000.npy'.format(value = value, parameter = parameter), allow_pickle=True)
        num_steps_logs.append(num_steps_log)
        num_unsolved_log = np.load('results/sensitivity_analysis/{parameter}/{parameter}{value}/num_unsolved_log_N3_Einf_GAMMA{value}_R_FINAL-10000.npy'.format(value = value, parameter = parameter))
        num_unsolved_logs.append(num_unsolved_log)
        V_log = np.load('results/sensitivity_analysis/{parameter}/{parameter}{value}/V_log_N3_Einf_GAMMA{value}_R_FINAL-10000.npy'.format(value = value, parameter = parameter))
        V_logs.append(V_log)

        #Plot error history 
        plt.figure('error')
        plt.plot(error_log, linewidth = 5, label = '{parameter} = {value}'.format(parameter = parameter, value = value))
        plt.xlabel('Step number [-]')
        plt.ylabel('Convergence error [-]')
        plt.grid()

        #Plot history of value function 
        plt.figure('V_{value}'.format(value = value))
        plot.plot_value_function(V_log, np.math.factorial(9))
        figure = plt.gcf()
        figure.set_size_inches(12, 10)
        plt.xlim((-20000,390000))
        plt.savefig('results/sesitivity_analysis/{parameter}/plots/V_{value}.png'.format(value = value, parameter = parameter))

        #Plot history of unsolved puzzles for each parameter 
        plt.figure('number unsolved puzzles')
        plt.bar(np.arange(len(num_unsolved_log)), num_unsolved_log, label = '{parameter} = {value}'.format(parameter = parameter, value = value), zorder = order_GAMMA[index], width = width_GAMMA[index])
        plt.xlabel('Number of outer loop [-]')
        plt.ylabel('Number of unsolved puzzles [-]')
        plt.grid()
        

        #Plot history of deadend for each parameter 
        plt.figure('number deadends')
        plt.bar(np.arange(len(num_deadends_log)), num_deadends_log, label = '{parameter} = {value}'.format(parameter = parameter, value = value), zorder = order_GAMMA[index], width = width_GAMMA[index])
        plt.xlabel('Number of outer loop [-]')
        plt.ylabel('Number of deadends [-]')
        plt.ylim((0,87000))
        plt.grid()

        #Plot history required steps for each parameter 
        means = [np.mean(num_steps) for num_steps in num_steps_log]
        plt.figure('required steps')
        plt.bar(np.arange(len(num_steps_log)), means, label = '{parameter} = {value}'.format(parameter = parameter, value = value), zorder = order_GAMMA[index], width = width_GAMMA[index])
        plt.xlabel('Number of outer loop [-]')
        plt.ylabel('Mean of required steps to solution [-]')
        plt.grid()
        plt.ylim((0,35))




    #Add legends to error history 
    plt.figure('error')
    plt.legend()
    figure = plt.gcf()
    figure.set_size_inches(12, 10)
    plt.savefig('results/sensitivity_analysis/{parameter}/plots/error.pdf'.format(parameter = parameter))


    plt.figure('number unsolved puzzles')
    plt.legend ()
    figure = plt.gcf()
    figure.set_size_inches(12, 10)
    plt.savefig('results/sensitivity_analysis/{parameter}/plots/unsolved.pdf'.format(parameter = parameter))

    plt.figure('number deadends')
    plt.legend()
    figure = plt.gcf()
    figure.set_size_inches(12, 10)
    plt.savefig('results/sensitivity_analysis/{parameter}/plots/deadends.pdf'.format(parameter = parameter))

    plt.figure('required steps')
    #plt.plot(means[-1]*np.ones(len(means)), 'k--', linewidth = 0.8, label = 'Converged value')
    figure = plt.gcf()
    figure.set_size_inches(12, 10)
    plt.legend()
    plt.savefig('results/sensitivity_analysis/{parameter}/plots/steps.pdf'.format(parameter = parameter))




    #Mean of number of steps required to arrive at solution at final iteration
    plt.figure('final_means')
    means = [np.mean(num_steps_log[-1]) for num_step_log in num_steps_logs]
    plt.bar(np.arange(len(E)), means)
    plt.xlabel('Error threshold [-]')
    plt.ylabel('Mean number of steps to arrive at solution [-]')
    plt.xticks(ticks = (0,1,2,3,4),labels = ('0.1', '5', '10', '20', 'inf'))
    plt.grid()


    #plt.show()



E = [0.1, 5, 10, 20, np.inf]
GAMMA = [0, 0.8, 0.9, 1, 1.5]
R_FINAL = [0, -100, -1000, -10000, -100000]

#plot_sensitivty_analysis(E, 'E')
plot_sensitivty_analysis(GAMMA, 'GAMMA')


