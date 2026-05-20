import numpy as np
import random as rand
from multiprocessing import Pool


class GA_paretto:

    def __init__(self, bounds, corrected_bounds, fitnessFunction_1, fitnessFunction_2, populationSize=10, mutationRate=0.07,
                 percent_to_preserve=0.5, island_model=False, graph=False, ui_plot=True):

        self.bounds = bounds
        self.corrected_bounds = corrected_bounds
        self.fitnessFunction_1 = fitnessFunction_1
        self.fitnessFunction_2 = fitnessFunction_2
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
        p = Pool(4)
        self.total_generations = 25
        self.current_generation = -1

        # visuals
        self.scale = 1
        self.graph = graph
        self.ui_plot = ui_plot

    def run(self):
        # get fitness for each individual
        print('compute fitness')

        self.current_generation += 1

        # compute fitness
        self.sumFitness = 0
        self.fitness_1_raw = p.map(self.computeFitness_1, self.indexes)
        self.fitness_2_raw = p.map(self.computeFitness_2, self.indexes)
        self.fitness_1 = list()
        self.fitness_2 = list()

        max_cost = max(self.fitness_1_raw)
        for j in range(len(self.fitness_1_raw)):
            self.fitness_1.append(max_cost - self.fitness_1_raw[j])

        max_cost = max(self.fitness_2_raw)
        for j in range(len(self.fitness_2_raw)):
            self.fitness_2.append(max_cost - self.fitness_2_raw[j])

        # start new population
        if not self.islands_model:
            self.reproduction_steps()

        self.fitnessFunction_1(self.best, plot=True)

    def reproduction_steps(self):
        print('new population')
        self.newPopulation = list()

        # breed new selecting parents based on roulette wheel selection
        print('breed')
        self.reproduce_paretto()

        # mutate individuals except for previous best
        print('mutate')
        for j in self.newPopulation:
            if j is not self.best:
                self.mutate(j)

        # set newPopulation as actual population
        self.population = self.newPopulation

    def reproduce_paretto(self):
        print('reproduce paretto')

        self.probabilities = list()
        for i in range(len(self.population)):
            dominated_by = 0
            for j in range(len(self.population)):
                if i == j:
                    continue
                if self.fitness_1[i] < self.fitness_1[j] and self.fitness_2[i] < self.fitness_2[j]:
                    dominated_by += 1
            self.probabilities.append(dominated_by)

        # find best
        self.bestScore = min(self.probabilities)
        self.best_index = self.probabilities.index(self.bestScore)
        self.best = self.population[self.best_index]
        self.newPopulation.append(self.best)
        self.bestScore = self.fitness_1[self.best_index]

        # reverse probabilities
        max_prob = max(self.probabilities)
        for i in range(len(self.probabilities)):
            self.probabilities[i] = max_prob - self.probabilities[i]

        sum_of_probabilities = 0
        for i in self.probabilities:
            sum_of_probabilities += i

        new_probabilities = list()
        new_probabilities.append(self.probabilities[self.best_index])
        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
            pRand = rand.uniform(0, sum_of_probabilities)
            total = 0;
            for i in range(len(self.population)):
                total += self.probabilities[i]
                if total >= pRand:
                    self.newPopulation.append(self.population[i])
                    new_probabilities.append(self.probabilities[i])
                    break;

        sum_of_probabilities = 0;
        for i in new_probabilities:
            sum_of_probabilities += i

        while len(self.newPopulation) < len(self.population):
            parent1 = None
            parent2 = None

            pRand = rand.uniform(0, sum_of_probabilities)
            total = 0;
            for i in range(len(self.newPopulation)):
                total += new_probabilities[i]
                if total >= pRand:
                    parent1 = self.newPopulation[i]
                    break

            pRand = rand.uniform(0, sum_of_probabilities)
            total = 0;
            for i in range(len(self.newPopulation)):
                total += new_probabilities[i]
                if total >= pRand:
                    parent2 = self.newPopulation[i]
                    break

            self.newPopulation.append(self.breed(parent1, parent2))

        print(self.probabilities)
        print('fitnesses: {}, {}, {} - best {}'.format(self.fitness_1_raw[self.population.index(self.best)],
                                                       self.fitness_2_raw[self.population.index(self.best)],
                                                       self.probabilities[self.population.index(self.best)], self.best))

    def reproduce_paretto_reverse(self):
        print('reproduce paretto reverse')

        self.probabilities = list()
        for i in range(len(self.population)):
            dominates = 0
            for j in range(len(self.population)):
                if i == j:
                    continue
                if self.fitness_1[i] > self.fitness_1[j] and self.fitness_2[i] > self.fitness_2[j]:
                    dominates += 1
            self.probabilities.append(dominates)

        print('probabilities: {}'.format(self.probabilities))

        # find best
        self.bestScore = max(self.probabilities)
        self.best_index = self.probabilities.index(self.bestScore)
        self.best = self.population[self.best_index]
        self.newPopulation.append(self.best)
        self.bestScore = self.fitness_1[self.best_index]

        sum_of_probabilities = 0
        for i in self.probabilities:
            sum_of_probabilities += i

        new_probabilities = list()
        new_probabilities.append(self.probabilities[self.best_index])
        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
            pRand = rand.uniform(0, sum_of_probabilities)
            total = 0;
            for i in range(len(self.population)):
                total += self.probabilities[i]
                if total >= pRand:
                    self.newPopulation.append(self.population[i])
                    new_probabilities.append(self.probabilities[i])
                    break;

        sum_of_probabilities = 0;
        for i in new_probabilities:
            sum_of_probabilities += i

        while len(self.newPopulation) < len(self.population):
            parent1 = None
            parent2 = None

            pRand = rand.uniform(0, sum_of_probabilities)
            total = 0;
            for i in range(len(self.newPopulation)):
                total += new_probabilities[i]
                if total >= pRand:
                    parent1 = self.newPopulation[i]
                    break

            pRand = rand.uniform(0, sum_of_probabilities)
            total = 0;
            for i in range(len(self.newPopulation)):
                total += new_probabilities[i]
                if total >= pRand:
                    parent2 = self.newPopulation[i]
                    break

            self.newPopulation.append(self.breed(parent1, parent2))

        print(self.probabilities)
        print('fitnesses: {}, {}, {} - best {}'.format(self.fitness_1_raw[self.population.index(self.best)],
                                                       self.fitness_2_raw[self.population.index(self.best)],
                                                       self.probabilities[self.population.index(self.best)], self.best))

    def computeFitness_1(self, i):
        f = self.fitnessFunction_1(self.population[i])
        self.fitness_1[i] = f
        print('fitness: {} + individual: {} + i:{}'.format(f, self.population[i], i))
        print()
        return f

    def computeFitness_2(self, i):
        f = self.fitnessFunction_2(self.population[i])
        self.fitness_2[i] = f
        print('fitness: {} + individual: {} + i:{}'.format(f, self.population[i], i))
        print()
        return f

    def breed(self, parent1, parent2):
        child = list()
        random1 = rand.randint(0, len(parent1) - 1)
        random2 = rand.randint(0, len(parent1) - 1)
        if random1 < random2:
            child.extend(parent2[:random1])
            child.extend(parent1[random1: random2])
            child.extend(parent2[random2:])
        else:
            child.extend(parent1[:random2])
            child.extend(parent2[random2: random1])
            child.extend(parent1[random1:])
        return child

    def mutate(self, individual):
        for i in range(len(individual)):
            if rand.uniform(0, 1) < self.mutationRate:
                if rand.uniform(0, 1) < 0.8:
                    if rand.uniform(0, 1) < 0.5:
                        individual[i] = individual[i] + individual[i] * 0.1
                    else:
                        individual[i] = individual[i] - individual[i] * 0.1
                else:
                    individual[i] = rand.uniform(self.bounds[i, 0], self.bounds[i, 1])
