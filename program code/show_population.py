import numpy as np
import time
import random as rand
from multiprocessing import Pool


class show_population:

    def __init__(self, bounds, fitnessFunction_1, populationSize=10, mutationRate=0.07,
                 percent_to_preserve=0.5, island_model=False, graph=False, ui_plot=True):

        self.bounds = bounds
        self.fitnessFunction_1 = fitnessFunction_1
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.percent_to_preserve = percent_to_preserve
        self.islands_model = island_model

        # GA lists
        self.population = list()
        self.newPopulation = list()

        for i in range(populationSize):
            l = np.random.uniform(bounds[:, 0], bounds[:, 1], 10)
            l2 = [y for y in l]
            self.population.append(l2)

        self.fitness_1 = list([0] * populationSize)
        self.fitness_2 = list([0] * populationSize)

        # GA breeding
        self.sumFitness = 0
        self.indexes = list()
        for i in range(populationSize):
            self.indexes.append(i)

        self.best = None
        self.bestScore = 0

        # simulation
        global p
        p = Pool(1)
        self.total_generations = 25
        self.current_generation = -1

        # visuals
        self.scale = 1
        self.graph = graph
        self.ui_plot = ui_plot

    def run(self):
        # get fitness for each individual
        print('show population')
        self.population.clear()
        for i in range(self.populationSize):
            l = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 10)
            l2 = [y for y in l]
            self.population.append(l2)

        self.current_generation += 1

        # compute fitness
        self.sumFitness = 0
        self.fitness_1_raw = p.map(self.computeFitness_1, self.indexes)

        self.bestScore = min(self.fitness_1_raw)
        self.best_index = self.fitness_1_raw.index(self.bestScore)
        self.best = self.population[self.best_index]
        self.bestScore = self.fitness_1[self.best_index]

        self.fitnessFunction_1(self.best, plot=True)

    def computeFitness_1(self, i):
        f = self.fitnessFunction_1(self.population[i])
        self.fitness_1[i] = f
        print('fitness: {} + individual: {} + i:{}'.format(f, self.population[i], i))
        print()
        return f
