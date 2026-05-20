from multiprocessing.pool import Pool

import numpy as np
from GA_base import GA_base
import random as rand

from co_evolution import sigma_GA


class fddc:
    def __init__(self, bounds, fitnessFunction, distribution_comparison, populationSize=10, fddc=True,
                 co_evolution=True, read_from_file=False, read_from_file_name="sim_val0", graph=False, ui_plot=True):

        # number of processes to be used. the higher the number the more
        # simulations can be run at once increasing the algorithms speed
        global p
        p = Pool(6)

        # GA parameters
        self.bounds = bounds
        self.fitnessFunction = fitnessFunction
        self.distribution_comparison = distribution_comparison
        self.populationSize = populationSize
        self.fddc = fddc # turning this off will turn the algorithm into regular co-evolution
        self.co_evolution = co_evolution # turning this and fddc off will turn algorithm into regular GA

        self.generation = -1
        self.points_to_distribute = 4 # for co-evolution how many points to add to a sigma value
        self.points_to_add = 4 # for co-evolution how many points to increase out of all sigmas
        self.memory_size = 10 if self.co_evolution else 1 # memory size for fddc and co-evolution
        self.number_of_children = 2 # number of children per generation
        self.number_of_encounters = 10 if self.co_evolution else 1 # number of encounters per generation
        self.original_sigma = list()
        for i in self.distribution_comparison.sigma:
            self.original_sigma.append(i)

        # if true then fake data will be loaded
        if read_from_file:
            print('read from file')
            self.distribution_comparison.set_exp_val(
                np.asarray(self.read_from_file(read_from_file_name)))

        self.best = None

        # used for encounters
        self.indexes = list()
        for i in range(populationSize):
            self.indexes.append(i)

        # lists that store the different fitness values per individual
        self.fitness_memory_pop1 = list()
        self.fitness_memory_pop2 = list()
        # list of individuals for both populations
        self.pop1 = list()
        self.pop2 = list()
        # rank list for both populations
        self.rank_pop1 = list()
        self.rank_pop2 = list()
        # probability for each rank to be chosen
        self.probabilities = list()
        for i in range(populationSize):
            self.probabilities.append(((i + 1) / self.populationSize) ** 1.5)

        # initialize populations
        for i in range(populationSize):
            l = np.random.uniform(bounds[:, 0], bounds[:, 1], 10)
            l2 = [y for y in l]
            if rand.uniform(0, 1) < 0.5:
                l2[len(l2) - 1] = 0.0
            else:
                l2[len(l2) - 1] = 1.0
            self.pop1.append(l2)

        print('pop1: {}'.format(self.pop1))

        # elongate sigma for more control to give fddc and co-evolution more control
        self.fitnessFunction(self.pop1[0])
        self.original_sigma = [1] * len(self.distribution_comparison.non_zero_indices[0])
        self.points_to_distribute = int(0.2 * len(self.original_sigma))
        print('elongated sigma: {}'.format(len(self.original_sigma)))
        print(self.original_sigma)

        # split the populations such that each individual has necassary number of encounters
        # ensure that population size % memory size == 0
        if self.populationSize % self.memory_size is not 0:
            self.memory_size = 1
        if self.co_evolution:
            for i in range(populationSize):
                individual = list()
                for j in range(len(self.original_sigma)):
                    individual.append(self.original_sigma[j])
                indices = list()
                while len(indices) < self.points_to_distribute:
                    location = rand.randint(0, len(self.original_sigma) - 1)
                    if location not in indices:
                        individual[location] += self.points_to_add
                        indices = [i for i in range(len(individual)) if individual[i] not in self.original_sigma]
                self.pop2.append(individual)
        else:
            for i in range(populationSize):
                individual = self.original_sigma
                self.pop2.append(individual)

        print('pop2: {}'.format(self.pop2))

        for i in range(self.populationSize):
            self.fitness_memory_pop1.append(list())
            self.fitness_memory_pop2.append(list())

        # initalize fitnesses
        self.second_indexes = list()
        multiplier = 0
        for i in range(populationSize):
            indi = list()
            if i is not 0:
                if i % self.memory_size is 0:
                    multiplier += 1
            for j in range(self.memory_size):
                indi.append(int(j + self.memory_size * multiplier))
            self.second_indexes.append(indi)
        print('si: {}'.format(self.second_indexes))

        costs = p.map(self.compute_initial_fitness, self.indexes)
        for i in range(len(costs)):
            for j in range(len(costs[0])):
                self.fitness_memory_pop1[i].append(-1 * costs[i][j])
                self.fitness_memory_pop2[self.second_indexes[i][j]].append(costs[i][j])

        print('fm1: {}', format(self.fitness_memory_pop1))
        print('fm2: {}', format(self.fitness_memory_pop2))

        # visuals
        self.scale = 1
        self.graph = graph
        self.ui_plot = ui_plot

    def read_from_file(self, file_name):
        file = open(file_name, 'r')
        text = file.read()
        text_array = text.split('/')
        to_return = list()
        for i in range(len(text_array) - 1):
            to_return.append(float(text_array[i]))
        return to_return

    def run(self):
        print('start fddc')
        self.generation += 1

        if self.generation is not 0 and self.co_evolution:
            self.current_indexes_pop1 = list()
            self.current_indexes_pop2 = list()

            # select individuals to have encounters
            # individuals pop 1
            print('probabilities: {}'.format(self.probabilities))
            print('sum: {}'.format(np.sum(self.probabilities)))

            for i in range(self.number_of_encounters):
                r = rand.uniform(0, np.sum(self.probabilities))
                s = 0
                for j in range(len(self.probabilities)):
                    s += self.probabilities[j]
                    if s >= r:
                        self.current_indexes_pop1.append(self.rank_pop1[j])
                        break

            # individuals pop 2
            for i in range(self.number_of_encounters):
                r = rand.uniform(0, np.sum(self.probabilities))
                s = 0
                for j in range(len(self.probabilities)):
                    s += self.probabilities[j]
                    if s >= r:
                        self.current_indexes_pop2.append(self.rank_pop2[j])
                        break

            # run encounters
            costs = p.map(self.computeFitness, [x for x in range(self.number_of_encounters)])
            for i in range(len(costs)):
                self.fitness_memory_pop1[self.pop1.index(self.current_indexes_pop1[i])].append(-1 * costs[i])
                self.fitness_memory_pop1[self.pop1.index(self.current_indexes_pop1[i])].pop(0)
                self.fitness_memory_pop2[self.pop2.index(self.current_indexes_pop2[i])].append(costs[i])
                self.fitness_memory_pop2[self.pop2.index(self.current_indexes_pop2[i])].pop(0)

        # provides ranks to all individuals
        # reproduce individual(s) for each population
        for i in range(self.number_of_children):
            self.rank_pop1 = list()
            self.rank_pop2 = list()
            self.compute_rank()
            self.reproduce_pop1()
            if self.co_evolution:
                self.reproduce_pop2()

        self.rank_pop1 = list()
        self.rank_pop2 = list()
        self.compute_rank()

        # select best individual
        self.best = self.rank_pop1[len(self.rank_pop1) - 1]
        self.distribution_comparison.set_sigma(self.original_sigma)

        for i in self.rank_pop1:
            print(np.sum(self.fitness_memory_pop1[self.pop1.index(i)]) / self.memory_size)

        print('best individual: {}'.format(self.best))
        self.best_score = self.fitnessFunction(self.best, plot=True)[0]
        print('best score: {}'.format(self.best_score))


    def compute_rank(self):
        # provide ranks pop 1
        found_sum_pop1 = list()
        minimum = None
        while len(self.rank_pop1) < len(self.pop1):
            for j in range(len(self.fitness_memory_pop1)):
                sum_fitness = np.sum(self.fitness_memory_pop1[j])
                if minimum is None and sum_fitness not in found_sum_pop1:
                    minimum = sum_fitness
                if minimum is not None:
                    if minimum > sum_fitness and sum_fitness not in found_sum_pop1:
                        minimum = sum_fitness
            if minimum is None:
                minimum = 0
            found_sum_pop1.append(minimum)
            copies = [self.pop1[i] for i in range(len(self.fitness_memory_pop1)) if
                      np.sum(self.fitness_memory_pop1[i]) == minimum]
            self.rank_pop1.extend(copies)
            minimum = None

        print('rank_pop1: {}'.format(self.rank_pop1))

        # provide ranks pop 2
        found_sum_pop2 = list()
        minimum = None
        while len(self.rank_pop2) < len(self.pop2):
            for j in range(len(self.fitness_memory_pop2)):
                sum_fitness = np.sum(self.fitness_memory_pop2[j])
                if minimum is None and sum_fitness not in found_sum_pop2:
                    minimum = sum_fitness
                if minimum is not None:
                    if minimum > sum_fitness and sum_fitness not in found_sum_pop2:
                        minimum = sum_fitness
            if minimum is None:
                minimum = 0
                found_sum_pop2.append(minimum)
            copies = [self.pop2[i] for i in range(len(self.fitness_memory_pop2)) if
                      np.sum(self.fitness_memory_pop2[i]) == minimum]
            found_sum_pop2.append(minimum)
            self.rank_pop2.extend(copies)
            minimum = None

        print('rank_pop2: {}'.format(self.rank_pop2))
        if not self.fddc:
            return

        # provide an adjusted ranking based on its novelty pop 2
        novelty_pop2 = list()
        for i in range(len(self.rank_pop2)):
            if i is 0:
                value = abs(np.sum(self.fitness_memory_pop2[self.pop2.index(self.rank_pop2[i])]) - np.sum(
                    self.fitness_memory_pop2[self.pop2.index(self.rank_pop2[i + 1])]))
            elif i is len(self.rank_pop2) - 1:
                value = abs(np.sum(self.fitness_memory_pop2[self.pop2.index(self.rank_pop2[i])]) - np.sum(
                    self.fitness_memory_pop2[self.pop2.index(self.rank_pop2[i - 1])]))
            else:
                value1 = abs(np.sum(self.fitness_memory_pop2[self.pop2.index(self.rank_pop2[i])]) - np.sum(
                    self.fitness_memory_pop2[self.pop2.index(self.rank_pop2[i + 1])]))
                value2 = abs(np.sum(self.fitness_memory_pop2[self.pop2.index(self.rank_pop2[i])]) - np.sum(
                    self.fitness_memory_pop2[self.pop2.index(self.rank_pop2[i - 1])]))
                if value1 < value2:
                    value = value1
                else:
                    value = value2
            novelty_pop2.append(value)

        print('novelty: {}'.format(novelty_pop2))

        self.new_rank_pop2 = list()

        found_novelty = list()
        minimum = None
        while len(self.new_rank_pop2) < len(self.pop2):
            for j in range(len(novelty_pop2)):
                sum_fitness = novelty_pop2[j]
                if minimum is None and sum_fitness not in found_novelty:
                    minimum = sum_fitness
                if minimum is not None:
                    if minimum > sum_fitness and sum_fitness not in found_novelty:
                        minimum = sum_fitness
            if minimum is None:
                minimum = 0
            found_novelty.append(minimum)
            copies = [self.rank_pop2[i] for i in range(len(novelty_pop2)) if novelty_pop2[i] == minimum]
            self.new_rank_pop2.extend(copies)
            minimum = None

        print('new rank pop 2: {}'.format(self.new_rank_pop2))

    # compute fitness
    def computeFitness(self, i):
        individual = self.current_indexes_pop1[i]
        opponent = self.current_indexes_pop2[i]
        self.distribution_comparison.set_sigma(opponent)
        cost = self.fitnessFunction(individual)[0]
        return cost

    # compute fitness
    def compute_fitness_no_index(self, ind):
        r = rand.randint(0, self.populationSize - 1)
        self.distribution_comparison.set_sigma(self.pop2[r])
        cost = self.fitnessFunction(ind)[0]
        return cost

    # compute fitness
    def compute_fitness_sigma(self, ind):
        r = rand.randint(0, self.populationSize - 1)
        self.distribution_comparison.set_sigma(ind)
        cost = self.fitnessFunction(self.pop1[r])[0]
        return cost

    # compute fitness
    def compute_initial_fitness(self, i):
        all_cost = list()
        self.fitnessFunction(self.pop1[i])
        for j in self.second_indexes[i]:
            self.distribution_comparison.set_sigma(self.pop2[j])
            cost = self.distribution_comparison.compute_cost()[0]
            all_cost.append(cost)
        print('progress: {} / {}'.format(i, len(self.indexes)))
        print(all_cost)
        return all_cost

    # reproduce pop 1
    def reproduce_pop1(self):
        # choose parents
        p1 = None
        j1 = 0
        p2 = None
        j2 = 0

        # select parent 1
        r = rand.uniform(0, np.sum(self.probabilities))
        s = 0
        for j in range(len(self.probabilities)):
            s += self.probabilities[j]
            if s >= r:
                p1 = self.rank_pop1[j]
                j1 = j
                break

        # select parent 2
        r = rand.uniform(0, np.sum(self.probabilities))
        s = 0
        for j in range(len(self.probabilities)):
            s += self.probabilities[j]
            if s >= r:
                p2 = self.rank_pop1[j]
                j2 = j
                break

        print('parent1: {} {}'.format(j1, p1))
        print('parent2: {} {}'.format(j2, p2))

        # create child by two point crossover and mutation
        child = self.breed(p1, p2)

        print('child: {}'.format(child))

        # create child memory
        fitnesses = list()
        self.fitnessFunction(child)
        for i in range(self.memory_size):
            r = rand.randint(0, self.populationSize - 1)
            self.distribution_comparison.set_sigma(self.pop2[r])
            cost = -1 * self.distribution_comparison.compute_cost()[0]
            fitnesses.append(cost)

        # compare child with worst individual
        print('sum child: {}, sum worst: {}'.format(np.sum(fitnesses),
                                                    np.sum(
                                                        self.fitness_memory_pop1[self.pop1.index(self.rank_pop1[0])])))
        # if child is better than worst individual then add to population
        if np.sum(fitnesses) > np.sum(self.fitness_memory_pop1[self.pop1.index(self.rank_pop1[0])]):
            print('add child')
            self.fitness_memory_pop1.pop(self.pop1.index(self.rank_pop1[0]))
            self.fitness_memory_pop1.append(fitnesses)
            self.pop1.remove(self.rank_pop1[0])
            self.pop1.append(child)

        else:
            print('did not add child')

    # reprocude population 2
    def reproduce_pop2(self):
        # choose parents
        p1 = None
        p2 = None

        # select parent 1
        r = rand.uniform(0, np.sum(self.probabilities))
        s = 0
        for j in range(len(self.probabilities)):
            s += self.probabilities[j]
            if s >= r:
                p1 = self.new_rank_pop2[j] if self.fddc else self.rank_pop2[j]
                break

        # select parent 2
        r = rand.uniform(0, np.sum(self.probabilities))
        s = 0
        for j in range(len(self.probabilities)):
            s += self.probabilities[j]
            if s >= r:
                p2 = self.new_rank_pop2[j] if self.fddc else self.rank_pop2[j]
                break

        print('parent1: {}'.format(p1))
        print('parent2: {}'.format(p2))

        # create child using two point crossover and mutation
        child = self.breed_sigma(p1, p2)

        print('child: {}'.format(child))
        children = list()
        for i in range(self.memory_size):
            children.append(child)

        # create child memory
        fitnesses = p.map(self.compute_fitness_sigma, children)
        for i in range(len(fitnesses)):
            fitnesses[i] = fitnesses[i]

        # compare child to worst individual
        print('sum child: {}, sum worst: {}'.format(np.sum(fitnesses),
                                                    np.sum(
                                                        self.fitness_memory_pop2[
                                                            self.pop2.index(self.rank_pop2[0])])))

        # add child if better than worst individual
        if np.sum(fitnesses) > np.sum(self.fitness_memory_pop2[self.pop2.index(self.rank_pop2[0])]):
            print('add sigma child')

            self.fitness_memory_pop2.pop(self.pop2.index(self.rank_pop2[0]))
            self.fitness_memory_pop2.append(fitnesses)
            self.pop2.remove(self.rank_pop2[0])
            self.pop2.append(child)

        else:
            print('did not add sigma child')

    # breed function for population 1 using two point crossover
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
        self.mutate(child)
        return child

    # breed function for population 2
    def breed_sigma(self, parent1, parent2):
        child = list()
        for i in range(len(self.original_sigma)):
            child.append(self.original_sigma[i])

        # find points in sigma that are larger than 1 for both parents
        parent1_pos = [i for i in range(len(parent1)) if parent1[i] not in self.original_sigma]
        parent2_pos = [i for i in range(len(parent2)) if parent2[i] not in self.original_sigma]
        parent_pos = [x for x in parent1_pos]
        for x in parent2_pos:
            parent_pos.append(x)

        # randomly add indices from list of parent indices
        child_pos = list()
        while len(child_pos) < self.points_to_distribute:
            pos = parent_pos[rand.randint(0, len(parent_pos) - 1)]
            if pos not in child_pos:
                child_pos.append(pos)

        # increase the values of selected incdices
        for x in child_pos:
            child[x] += self.points_to_add

        return child

    # mutate function for population 1
    def mutate(self, individual):
        for i in range(len(individual) - 1):
            if i is len(individual) - 1:
                if rand.uniform(0, 1) < 0.5:
                    individual[i] = 0
                else:
                    individual[i] = 1
                break
            if rand.uniform(0, 1) < 0.6:
                if rand.uniform(0, 1) < 0.5:
                    individual[i] = individual[i] + individual[i] * 0.001
                    if individual[i] > self.bounds[i, 1]:
                        individual[i] = self.bounds[i, 1]
                else:
                    individual[i] = individual[i] - individual[i] * 0.001
                    if individual[i] < self.bounds[i, 0]:
                        individual[i] = self.bounds[i, 0]
            else:
                individual[i] = rand.uniform(self.bounds[i, 0], self.bounds[i, 1])
