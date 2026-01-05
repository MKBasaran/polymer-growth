# PolymerGrowthSimulator

Dependencies:
matplotlib
numpy
scipy
sklearn
multiprocessing

In order to use the GUI run 'python user_interface.py'. The usage of each section of the graphical interface is provided below:

1) Algorithm:
In the algorithm section you can choose which optimisation method  to select. Currently, two methods are supported namely Evolutionary Algorithm and Bayesian Optimisation methods.
Each algorithm can be tweaked according to the 
Number of iterations: The number of iterations, in simple words, how many times the algorithm is run. It basically decides how long it takes for the whole process to stop. A value between 25 to 30 is suggested for a small population size (10 - 30). 
Population size: For a faster convergence, a population size of 15 is recommended. But if time is not constrained a larger population size value between 80 to 100 should be used. Correspondingly, number of iterations should also be increased to 50 - 80.

2) Cost function:
A cost function can be chosen out of the three implemented ie, Min-max, median foldage and translational invariant. Median Fold Change and Translation Invariant functions are observed to give good results.
Min-Max is not recommended to be chosen.

3) Parameters:
filename:
Here, the file needs to be selected which contains the experimental data, which the simulation needs to match with.

4) Initial Parameters
Here the range of each simulation parameter needs to specified. The algorithm initiates with the mean value of the parameter and makes sure the bounds are followed.
A description of each parameter is mentioned below:
number_of_molecules - the number of starting chains (length 1)
time_sim - the number of timesteps the simulation runs for
p_growth - the probability of growth for each monomer, each time step
p_death - the probability of a monomer dying and joining the dead pool
p_dead_react - the probability a dead polymer reacts with a living one
kill_spawns_new - a binary flag as to whether a kill event means a new polymer of length 1 is spawned or not 1=killing event spawns a new polymer chain, 0 means it doesn't. 
monomer_pool - the size of the monomer pool, if it is a negative number then no monomer pool is used and monomer supply is assumed infinite, p_growth is now p_growth as provided * monomer_pool size
l_exponent - the exponent of the living length in the probability calculation for successful coupling
d_exponent - the exponent of the dead length in the probability
If upper limit is same as lower limit, the algorithm will result simulation results without any optimization.

5) Result Screen:
There are three distributions shows here. The first corresponds to the experimental data, the second shows the live the simulation results as it updates after each iteration, and the third shows the best match found after applying the normalization

6) Simulation Tab: 
Here, all the simulation parameters mentioned above can be mentioned and the simulation is run without any optimisation methods. 

