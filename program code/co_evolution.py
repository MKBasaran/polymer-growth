import sys

import numpy as np
from GA_base import GA_base
import random as rand

from GA_base_NN import GA_base_NN


class co_evolution:
    def __init__(self, bounds, corrected_bounds, fitnessFunction, distribution_comparison, populationSize=10,
                 mutationRate=0.05,
                 migration_number=10, number_of_islands=2, read_from_file=False, read_from_file_name="sim_val0",
                 graph=False, ui_plot=True):
        # GA parameters
        self.bounds = bounds
        self.corrected_bounds = corrected_bounds
        self.fitnessFunction = fitnessFunction
        self.distribution_comparison = distribution_comparison
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.migration_number = migration_number
        if migration_number * number_of_islands >= populationSize - 1:
            self.migration_number = int(populationSize / number_of_islands - 2)
        if self.migration_number < 0:
            self.migration_number = 0
        self.generation = -1
        self.number_of_islands = number_of_islands
        self.points_to_distribute = 1
        self.original_sigma = list()
        for i in self.distribution_comparison.sigma:
            self.original_sigma.append(i)

        if read_from_file:
            print('read from file')
            self.distribution_comparison.set_exp_val(
                np.asarray(self.read_from_file(read_from_file_name)))  # set before normalization

        # corrected_sigma = list()
        # to_add = 30 / len(self.fitness_class.sigma)
        # for i in range(len(self.fitness_class.sigma)):
        #     corrected_sigma.append(self.fitness_class.sigma[i] + to_add)
        # self.fitness_class.set_sigma(corrected_sigma)

        # standard
        # for i in range(len(self.fitness_class.sigma)):
        #     self.original_sigma.append(10)

        # sum_of_sigma = np.sum(self.fitness_class.sigma)
        # for i in range(len(self.fitness_class.sigma)):
        #     self.fitness_class.sigma[i] = self.fitness_class.sigma[i] + self.fitness_class.sigma[i] / (
        #                 sum_of_sigma / self.points_to_distribute)

        # start GA's
        self.algorithms = list()
        self.algorithms.append(
            GA_base(bounds, fitnessFunction, self.distribution_comparison, populationSize, selection_function=0,
                    island_model=False, co_evolution=True, read_from_file=read_from_file, read_from_file_name=read_from_file_name))
        self.algorithms.append(
            (sigma_GA(self.original_sigma, distribution_comparison=self.distribution_comparison,
                      points_to_distribute=self.points_to_distribute)))
        self.best = None

        # visuals
        self.scale = 1
        self.graph = graph
        self.ui_plot = ui_plot

    def run(self):
        print('start co-evolution')
        self.generation += 1
        self.algorithms[0].run()

        # if ga_nn then best_paramenters
        self.best = self.algorithms[0].best
        self.best_score = self.algorithms[0].best_corrected_score

        # run sigma GA
        if self.generation % 3 == 0:
            self.algorithms[1].run(self.best)

        print('sigma: {}'.format(self.algorithms[1].best))
        print('best individual: {}'.format(self.best))
        print('best corrected score: {}'.format(self.best_score))
        for i in range(len(self.algorithms)):
            print('best algorithm: {}: {}'.format(i, self.algorithms[i].bestScore))
        print()

        self.fitnessFunction(self.best, plot=True)

    def read_from_file(self, file_name):
        file = open(file_name, 'r')
        text = file.read()
        text_array = text.split('/')
        to_return = list()
        for i in range(len(text_array) - 1):
            to_return.append(float(text_array[i]))
        return to_return


class sigma_GA:
    def __init__(self, original_sigma, population_size=20
                 , distribution_comparison=None, points_to_distribute=1, points_to_add=2):
        print('start sigma_GA')
        self.distribution_comparison = distribution_comparison
        self.population = list()

        self.percent_to_preserve = 0.5

        self.total_generation = 30
        self.current_generation = 0
        self.original_sigma = original_sigma

        self.points_to_distribute = points_to_distribute
        self.points_to_add = points_to_add

        self.fitness = list()
        # for i in range(population_size):
        #     individual = list()
        #     for j in range(len(fitness_function.sigma)):
        #         individual.append(rand.uniform(0, 1))
        #     sum_individual = np.sum(individual)
        #     for j in range(len(individual)):
        #         individual[j] = self.original_sigma[j] + (individual[j] / (sum_individual / self.points_to_distribute))
        #     self.population.append(individual)

        for i in range(population_size):
            individual = list()
            for j in range(len(self.original_sigma)):
                individual.append(self.original_sigma[j])
            indices = list()
            while len(indices) < self.points_to_distribute:
                location = rand.randint(0, len(self.original_sigma) - 1)
                if location not in indices:
                    individual[location] += self.points_to_add
                    indices = [i for i in range(len(individual)) if individual[i] not in self.original_sigma]
            self.population.append(individual)

        self.best = None
        self.bestScore = 0

    def run(self, other_best):
        # compute fitness
        self.current_generation += 1
        print('run sigma GA')
        self.fitness = list()
        self.newPopulation = list()
        self.distribution_comparison.set_sigma([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        original_f = self.distribution_comparison.costFunction(other_best)

        for i in range(len(self.population)):
            print('individual: {}'.format(self.population[i]))
            self.distribution_comparison.set_sigma(self.population[i])
            f = self.distribution_comparison.compute_cost()
            self.fitness.append(f)
            print('{} : {}'.format(f, self.population[i]))

        print('original f: {}'.format(original_f))

        max_value = np.max(self.fitness)

        self.bestScore = max_value
        self.best = self.population[self.fitness.index(max_value)]
        self.newPopulation.append(self.best)

        # reproduce
        self.reproduce_tournament()

        for i in self.newPopulation:
            if i is not self.best:
                self.mutate(i)

        # for i in range(len(self.population)):
        #     print('{} + {}'.format(self.fitness[i], self.population[i]))

        self.population = self.newPopulation
        self.distribution_comparison.set_sigma(self.best)

    def reproduce_clean_slate(self, n=2):
        print('reproduce clean slate')
        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
            competitors = list()
            for i in range(n):
                competitor = rand.randint(0, len(self.population) - 1)
                while competitor in competitors:
                    competitor = rand.randint(0, len(self.population) - 1)
                competitors.append(competitor)

            best_fitness = 0
            best_competitor = None
            for i in competitors:
                if best_fitness == 0:
                    best_fitness = self.fitness[i]
                    best_competitor = i
                if best_fitness < self.fitness[i]:
                    best_fitness = self.fitness[i]
                    best_competitor = i
            winner = self.population[best_competitor]
            self.newPopulation.append(winner)

        while len(self.newPopulation) < len(self.population):
            individual = list()
            for j in range(len(self.original_sigma)):
                individual.append(self.original_sigma[j])
            indices = list()
            while len(indices) < self.points_to_distribute:
                location = rand.randint(0, len(self.original_sigma) - 1)
                if location not in indices:
                    individual[location] += self.points_to_add
                    indices = [i for i in range(len(individual)) if individual[i] not in self.original_sigma]
            self.population.append(individual)

    def reproduce_boltzmann(self, alpha=0.05, start_temperature=20):
        print('reproduce boltzmann')
        # create probabilities
        probabilities = list()

        # boltzmann probability
        f_max = max(self.fitness)
        k = (1 + 100 * self.current_generation / self.total_generation)
        temperature = start_temperature * (1 - alpha) ** k

        # compute probabilities
        for i in range(len(self.fitness)):
            probabilities.append(np.exp(-1 * ((f_max - self.fitness[i]) / temperature)))

        print('boltzmann probabilities: {}'.format(probabilities))
        # select individuals to carry over to next generation
        sum_of_probabilities = 0
        for i in probabilities:
            sum_of_probabilities += i

        new_probabilities = list()
        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
            pRand = rand.uniform(0, sum_of_probabilities)
            sum = 0;
            for j in range(len(self.population)):
                sum += probabilities[j]
                if sum >= pRand:
                    self.newPopulation.append(self.population[j])
                    new_probabilities.append(probabilities[j])
                    break

        # breed new individuals
        sum_of_probabilities = 0
        for i in new_probabilities:
            sum_of_probabilities += i

        while len(self.newPopulation) < len(self.population):
            parent1 = None
            parent2 = None

            pRand = rand.uniform(0, sum_of_probabilities)
            sum = 0;
            for j in range(len(self.newPopulation)):
                sum += new_probabilities[j]
                if sum >= pRand:
                    parent1 = self.newPopulation[j]
                    break

            pRand = rand.uniform(0, sum_of_probabilities)
            sum = 0;
            for j in range(len(self.newPopulation)):
                sum += new_probabilities[j]
                if sum >= pRand:
                    parent2 = self.newPopulation[j]
                    break

            self.newPopulation.append(self.breed(parent1, parent2))

    def reproduce_roulette(self):
        # select individuals to copy and are potential parents
        print('reproduce roulette')
        adjusted_fitness_sum = 0
        new_fitness = list()
        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):

            pRand = rand.uniform(0, np.sum(self.fitness))

            sum = 0;
            for j in range(len(self.population)):
                sum += self.fitness[j]
                if sum >= pRand:
                    self.newPopulation.append(self.population[j])
                    new_fitness.append(self.fitness[j])
                    adjusted_fitness_sum += self.fitness[j]
                    break

        while len(self.newPopulation) < len(self.population):
            # select parents
            parent1 = None
            parent2 = None
            pRand = rand.uniform(0, adjusted_fitness_sum)

            sum = 0;
            for j in range(len(self.newPopulation)):
                sum += new_fitness[j]
                if j == len(self.newPopulation) - 1 and sum < pRand:
                    print('sump1: {} , pRand: {}'.format(sum, pRand))
                if sum >= pRand:
                    parent1 = self.newPopulation[j]
                    break

            pRand = rand.uniform(0, adjusted_fitness_sum)

            sum = 0;
            for j in range(len(self.newPopulation)):
                sum += new_fitness[j]
                if j == len(self.newPopulation) - 1 and sum < pRand:
                    print('sump2: {} , pRand: {}'.format(sum, pRand))
                if sum >= pRand:
                    parent2 = self.newPopulation[j]
                    break

            # breed child (its safe to add them right away because adjusted sum already limits search)
            self.newPopulation.append(self.breed(parent1, parent2))

    def reproduce_tournament(self, n=3):
        print('reproduce tournament')
        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
            competitors = list()
            for i in range(n):
                competitor = rand.randint(0, len(self.population) - 1)
                while competitor in competitors:
                    competitor = rand.randint(0, len(self.population) - 1)
                competitors.append(competitor)

            best_fitness = 0
            best_competitor = None
            for i in competitors:
                if best_fitness == 0:
                    best_fitness = self.fitness[i]
                    best_competitor = i
                if best_fitness < self.fitness[i]:
                    best_fitness = self.fitness[i]
                    best_competitor = i
            winner = self.population[best_competitor]
            self.newPopulation.append(winner)

        while len(self.newPopulation) < len(self.population):
            competitors = list()
            for i in range(n):
                competitor = rand.randint(0, len(self.newPopulation) - 1)
                while competitor in competitors:
                    competitor = rand.randint(0, len(self.newPopulation) - 1)
                competitors.append(competitor)

            best_fitness = 0
            best_competitor = None
            for i in competitors:
                if best_fitness == 0:
                    best_fitness = self.fitness[i]
                    best_competitor = i
                if best_fitness < self.fitness[i]:
                    best_fitness = self.fitness[i]
                    best_competitor = i
            parent1 = self.population[best_competitor]

            competitors = list()
            for i in range(n):
                competitor = rand.randint(0, len(self.newPopulation) - 1)
                while competitor in competitors:
                    competitor = rand.randint(0, len(self.newPopulation) - 1)
                competitors.append(competitor)

            best_fitness = 0
            best_competitor = None
            for i in competitors:
                if best_fitness == 0:
                    best_fitness = self.fitness[i]
                    best_competitor = i
                if best_fitness < self.fitness[i]:
                    best_fitness = self.fitness[i]
                    best_competitor = i
            parent2 = self.population[best_competitor]

            self.newPopulation.append(self.breed(parent1, parent2))

    def reproduce_rank(self, percent_to_preserve=0.5):
        # rank and probabilities are in the same order. use index later to find actual individual
        print('reproduce rank')

        probabilities = list()
        rank_probabilities_sum = 0
        rank = list()
        minimum = None
        while len(rank) < len(self.population):
            for j in range(len(self.fitness)):
                if minimum is None and self.fitness[j] not in rank:
                    minimum = self.fitness[j]
                if minimum is not None:
                    if minimum > self.fitness[j] and self.fitness[j] not in rank:
                        minimum = self.fitness[j]
            if minimum is None:
                minimum = 0
            copies = [x for x in self.fitness if x == minimum]
            rank.extend(copies)
            minimum = None

        for i in range(len(rank)):
            probabilities.append(rank[i] / (len(rank) * (len(rank) - 1)))

        for i in range(len(probabilities)):
            rank_probabilities_sum += probabilities[i]

        new_fitness = list()
        new_fitness.append(rank[len(rank) - 1])
        while len(self.newPopulation) < int(len(self.population) * percent_to_preserve):

            pRand = rand.uniform(0, rank_probabilities_sum)
            sum = 0;
            for j in range(len(rank)):
                sum += probabilities[j]
                if sum >= pRand:
                    self.newPopulation.append(self.population[self.fitness.index(rank[j])])
                    new_fitness.append(rank[j])
                    break

        probabilities = list()
        rank_probabilities_sum = 0
        rank = list()
        minimum = None
        while len(rank) < len(self.newPopulation):
            for j in range(len(new_fitness)):
                if minimum is None and new_fitness[j] not in rank:
                    minimum = new_fitness[j]
                if minimum is not None:
                    if minimum > new_fitness[j] and new_fitness[j] not in rank:
                        minimum = new_fitness[j]
            if minimum is None:
                minimum = 0
            copies = [x for x in new_fitness if x == minimum]
            rank.extend(copies)
            minimum = None

        for i in range(len(rank)):
            probabilities.append(rank[i] / (len(rank) * (len(rank) - 1)))

        for i in range(len(probabilities)):
            rank_probabilities_sum += probabilities[i]

        while len(self.newPopulation) < len(self.population):
            parent1 = None
            parent2 = None
            pRand = rand.uniform(0, rank_probabilities_sum)
            sum = 0;
            for j in range(len(probabilities)):
                sum += probabilities[j]
                if sum >= pRand:
                    parent1 = self.population[self.fitness.index(rank[j])]
                    break

            pRand = rand.uniform(0, rank_probabilities_sum)
            sum = 0;
            for j in range(len(probabilities)):
                sum += probabilities[j]
                if sum >= pRand:
                    parent2 = self.population[self.fitness.index(rank[j])]
                    break

            self.newPopulation.append(self.breed(parent1, parent2))

    def breed(self, parent1, parent2):
        child = list()
        for i in range(len(self.original_sigma)):
            child.append(self.original_sigma[i])

        parent1_pos = [i for i in range(len(parent1)) if parent1[i] not in self.original_sigma]
        parent2_pos = [i for i in range(len(parent2)) if parent2[i] not in self.original_sigma]
        parent_pos = [x for x in parent1_pos]
        for x in parent2_pos:
            parent_pos.append(x)

        child_pos = list()
        while len(child_pos) < self.points_to_distribute:
            pos = parent_pos[rand.randint(0, len(parent_pos) - 1)]
            if pos not in child_pos:
                child_pos.append(pos)

        for x in child_pos:
            child[x] += self.points_to_add

        return child

    def mutate(self, individual, mutationRate=0.05):
        for i in range(len(individual)):
            if rand.uniform(0, 1) < mutationRate:
                individual_pos = [i for i in range(len(individual)) if individual[i] not in self.original_sigma]
                pos = rand.randint(0, len(individual_pos) - 1)
                individual[individual_pos[pos]] -= self.points_to_add

                individual_pos = [i for i in range(len(individual)) if individual[i] not in self.original_sigma]
                pos = rand.randint(0, len(individual) - 1)
                while pos in individual_pos:
                    pos = rand.randint(0, len(individual) - 1)
                individual[pos] += self.points_to_add
