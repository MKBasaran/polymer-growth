import numpy as np
from GA_base import GA_base
import random as rand


class cross_island_breeding_GA:
    def __init__(self, bounds, corrected_bounds, fitnessFunction, populationSize=10, mutationRate=0.05,
                 migration_number=10, number_of_islands=2, graph=False, ui_plot=True):
        # GA parameters
        self.bounds = bounds
        self.corrected_bounds = corrected_bounds
        self.fitnessFunction = fitnessFunction
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.migration_number = migration_number
        if migration_number * number_of_islands >= populationSize - 1:
            self.migration_number = int(populationSize / number_of_islands - 2)
        if self.migration_number < 0:
            self.migration_number = 0
        self.generation = 0
        self.number_of_islands = number_of_islands

        # start GA's
        print('islands: {}'.format(number_of_islands))
        self.algorithms = list()
        for i in range(number_of_islands):
            if i == 0:
                print('i: {} value: 0'.format(i))
                self.algorithms.append(
                    GA_base(bounds, fitnessFunction, populationSize, selection_function=4, island_model=True))

            elif i == 1:
                print('i: {} value: 3'.format(i))
                self.algorithms.append(
                    GA_base(corrected_bounds, fitnessFunction, populationSize, selection_function=4, island_model=True))

        self.best = None

        # visuals
        self.scale = 1
        self.graph = graph
        self.ui_plot = ui_plot

    def run(self):
        self.generation += 1
        for a in self.algorithms:
            a.run()

        if self.generation % 10 == 0:
            self.cross_breeding()
        else:
            for i in self.algorithms:
                i.reproduction_steps()

        for i in range(len(self.algorithms)):
            print('best algorithm: {}: {}'.format(i, self.algorithms[i].bestScore))

        if self.algorithms[0].bestScore < self.algorithms[1].bestScore:
            self.best = self.algorithms[0].best
        else:
            self.best = self.algorithms[1].best

        self.fitnessFunction(self.best, plot=True)

    def cross_breeding(self):
        print('cross breeding roulette')
        # half regular and half cross

        old_population1 = self.algorithms[0].population
        old_population2 = self.algorithms[1].population

        old_fitness1 = self.algorithms[0].fitness
        old_fitness2 = self.algorithms[1].fitness

        sum_fitness1 = np.sum(old_fitness1)
        sum_fitness2 = np.sum(old_fitness2)

        new_population1 = list()
        new_population2 = list()
        new_population1.append(self.algorithms[0].best)
        new_population2.append(self.algorithms[1].best)

        new_fitness1 = list()
        new_fitness2 = list()
        new_fitness1.append(self.algorithms[0].fitness[self.algorithms[0].population.index(self.algorithms[0].best)])
        new_fitness2.append(self.algorithms[1].fitness[self.algorithms[1].population.index(self.algorithms[1].best)])

        # select individual to keep
        for i in range(int(len(old_population1) / 2)):
            rand_value = rand.uniform(0, sum_fitness1)
            sum = 0
            for j in range(len(old_population1)):
                sum += old_fitness1[j]
                if sum >= rand_value:
                    new_population1.append(old_population1[j])
                    new_fitness1.append(old_fitness1[j])
                    break

        for i in range(int(len(old_population2) / 2)):
            rand_value = rand.uniform(0, sum_fitness2)
            sum = 0
            for j in range(len(old_population2)):
                sum += old_fitness2[j]
                if sum >= rand_value:
                    new_population2.append(old_population2[j])
                    new_fitness2.append(old_fitness2[j])
                    break

        # reproduce
        sum_fitness1 = np.sum(new_fitness1)
        sum_fitness2 = np.sum(new_fitness2)

        children1 = list()
        children2 = list()

        # alg 1
        for i in range(int(len(new_population1) / 2)):
            parent1 = None
            parent2 = None

            rand_value = rand.uniform(0, sum_fitness1)
            sum = 0
            for j in range(len(new_population1)):
                sum += new_fitness1[j]
                if sum >= rand_value:
                    parent1 = new_population1[j]
                    break

            rand_value = rand.uniform(0, sum_fitness1)
            sum = 0
            for j in range(len(new_population1)):
                sum += new_fitness1[j]
                if sum >= rand_value:
                    parent2 = new_population1[j]
                    break

            children1.append(self.algorithms[0].breed(parent1, parent2))

        #alg 2
        for i in range(int(len(new_population2) / 2)):
            parent1 = None
            parent2 = None

            rand_value = rand.uniform(0, sum_fitness2)
            sum = 0
            for j in range(len(new_population2)):
                sum += new_fitness2[j]
                if sum >= rand_value:
                    parent1 = new_population2[j]
                    break

            rand_value = rand.uniform(0, sum_fitness2)
            sum = 0
            for j in range(len(new_population2)):
                sum += new_fitness2[j]
                if sum >= rand_value:
                    parent2 = new_population2[j]
                    break

            children2.append(self.algorithms[1].breed(parent1, parent2))

        # cross
        while len(children1) < len(new_population1):
            parent1 = None
            parent2 = None

            rand_value = rand.uniform(0, sum_fitness1)
            sum = 0
            for j in range(len(new_population1)):
                sum += new_fitness1[j]
                if sum >= rand_value:
                    parent1 = new_population1[j]
                    break

            rand_value = rand.uniform(0, sum_fitness2)
            sum = 0
            for j in range(len(new_population2)):
                sum += new_fitness2[j]
                if sum >= rand_value:
                    parent2 = new_population2[j]
                    break

            children1.append(self.algorithms[0].breed(parent1, parent2))
            children2.append(self.algorithms[1].breed(parent1, parent2))

        new_population1.extend(children1)
        new_population2.extend(children2)

        self.algorithms[0].population = new_population1
        self.algorithms[1].population = new_population2





