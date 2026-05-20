import numpy as np
import random as rand
from multiprocessing import Pool

import pyautogui

from GA_base import GA_base
from GA_base_NN import GA_base_NN
from co_evolution import co_evolution
from fddc import fddc
from island_GA import island_GA


class tester:

    def __init__(self, bounds, corrected_bounds, fitnessFunction, distribution_comparison=None, populationSize=10,
                 number_of_generations=50, mutationRate=1.0, percent_to_preserve=0.5,
                 selection_function=2, mutate_function=3, breed_function=2,
                 island_model=False, graph=False,
                 ui_plot=True, read_from_file=False, read_from_file_name="."):

        # GA parameters
        self.bounds = bounds
        self.corrected_bounds = corrected_bounds
        self.fitnessFunction = fitnessFunction
        self.distribution_comparison = distribution_comparison
        self.populationSize = populationSize
        self.mutationRate = mutationRate # is 1 since 60 percent is uniformily perturbed and 40 percent is random
        self.percent_to_preserve = percent_to_preserve
        self.number_of_generations = number_of_generations
        #  selection function values:
        #  0 = roulette wheel (point wise gene selection)(exploitation)
        #  1 = rank (slow exploitation)
        #  2 = tournament (exploration but depends on tournament size)
        #  3 = completely random (exploration)
        #  4 = boltzmann (slowly becomes exploitation)
        self.selection_function = selection_function
        #  mutate function values:
        #  1 = only one value at a time
        #  2 = every value can mutate at a time
        #  3 = every value can mutate and often will transform uniformly
        self.mutate_function = mutate_function
        #  breed function values:
        #  1 = point wise crossover
        #  2 = chunk crossover
        self.breed_function = breed_function
        self.islands_model = island_model
        self.read_from_file = read_from_file
        self.read_from_file_name = read_from_file_name

        # visuals
        self.scale = 1
        self.graph = graph
        self.ui_plot = ui_plot

        # fairly useless variables that could be used for multiple runs
        self.file_counter = 0

        self.all_results = list()
        self.best_individuals = list()
        self.best = None

        self.current_results = list()
        self.current_individuals = list()

        # which algorithm to use
        # 0 = island model
        # 1 = co-evolution (useless version)
        # 2 = basic genetic algorithm
        # 3 = basic genetic algorithm using NN encoding  (useless)
        # 4 = fitness diversity driven co-evolution
        self.option = 2

    # create instance of desired algorithm
    def create_GA(self):
        if self.option is 0:
            self.iga = island_GA(self.bounds, corrected_bounds=self.corrected_bounds,
                                 fitnessFunction=self.fitnessFunction,
                                 fitnessfunction2=self.fitnessFunction, distribution_comp=self.distribution_comparison,
                                 populationSize=self.populationSize,
                                 number_of_islands=4, graph=self.graph, ui_plot=self.ui_plot,
                                 read_from_file=self.read_from_file,
                                 read_from_file_name=self.read_from_file_name)
        if self.option is 1:
            self.ce = co_evolution(bounds=self.bounds, corrected_bounds=self.corrected_bounds,
                                   fitnessFunction=self.fitnessFunction,
                                   distribution_comparison=self.distribution_comparison,
                                   populationSize=self.populationSize,
                                   mutationRate=self.mutationRate, read_from_file=self.read_from_file,
                                   read_from_file_name=self.read_from_file_name)
        if self.option is 2:
            self.ga = GA_base(bounds=self.bounds,
                              fitnessFunction=self.fitnessFunction,
                              distribution_comparison=self.distribution_comparison,
                              populationSize=self.populationSize,
                              selection_function=0,
                              read_from_file=self.read_from_file,
                              read_from_file_name=self.read_from_file_name)
        if self.option is 3:
            self.ga_nn = GA_base_NN(bounds=self.bounds,
                                    fitnessFunction=self.fitnessFunction,
                                    distribution_comparison=self.distribution_comparison,
                                    populationSize=self.populationSize,
                                    mutationRate=self.mutationRate, read_from_file=self.read_from_file,
                                    read_from_file_name=self.read_from_file_name)

        if self.option is 4:
            self.fddc = fddc(bounds=self.bounds, fitnessFunction=self.fitnessFunction,
                             distribution_comparison=self.distribution_comparison,
                             populationSize=self.populationSize, read_from_file=self.read_from_file,
                             read_from_file_name=self.read_from_file_name)
    def run(self):
        # run desired algorithm
        if self.option is 0:
            self.iga.run()
            self.current_results.append(self.iga.best_score)
            self.best = self.iga.best
        if self.option is 1:
            self.ce.run()
            self.current_results.append(self.ce.best_score)
            self.best = self.ce.best
        if self.option is 2:
            self.ga.run()
            self.current_results.append(self.ga.bestScore)
            self.best = self.ga.best
        if self.option is 3:
            self.ga_nn.run()
            self.current_results.append(self.ga_nn.bestScore)
            self.best = self.ga_nn.best.compute_output(self.ga_nn.input)
        if self.option is 4:
            self.fddc.run()
            self.current_results.append(self.fddc.best_score)
            self.best = self.fddc.best

    # stores the results, mostly used for multiple runs
    def store_best(self):
        self.all_results = list()
        self.all_results.extend(self.current_results)
        self.best_individuals.extend(self.current_individuals)

        # myScreenshot = pyautogui.screenshot()
        # myScreenshot.save('graph{}.png'.format(j))

        print("all results")
        for i in range(len(self.all_results)):
            print('{}'.format(self.all_results[i]))
        self.current_results = list()
        self.distribution_comparison.lowest_cost = np.inf
