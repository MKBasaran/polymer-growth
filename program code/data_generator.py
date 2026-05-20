import numpy as np
import random as rand
from multiprocessing import Pool


class data_generator:

    def __init__(self, bounds, fitnessFunction, populationSize=10, graph=False, ui_plot=True):
        # GA parameters
        # print('bounds: {} : {}'.format(type(bounds), bounds))
        self.bounds = bounds
        self.fitnessFunction = fitnessFunction
        self.populationSize = populationSize

        # GA lists
        self.population = list()

        for i in range(populationSize):
            l = np.random.uniform(bounds[:, 0], bounds[:, 1], 10)
            l2 = [y for y in l]
            if rand.uniform(0, 1) < 0.5:
                l2[len(l2) - 1] = 0.0
            else:
                l2[len(l2) - 1] = 1.0
            self.population.append(l2)

        self.fitness = list([0] * populationSize)

        self.best = None
        self.bestScore = 0

        self.indexes = list()
        for i in range(populationSize):
            self.indexes.append(i)

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
        # print('compute fitness')

        self.current_generation += 1

        # compute fitness
        self.sumFitness = 0
        for i in range(len(self.population)):
            self.computeFitness(i)

        bestFit = min(self.fitness)
        self.bestScore = min(bestFit)
        self.best = self.population[self.fitness.index(bestFit)]
        print("best score: {}".format(self.bestScore))

        self.fitnessFunction(self.best, plot=True)

    def computeFitness(self, i):
        f = self.fitnessFunction(self.population[i])
        self.fitness[i] = f
        # print('fitness: {} + individual: {} + i:{}'.format(f, self.population[i], i))
        # print()
        return f