import numpy as np
import random as rand
import time
from multiprocessing import Pool
import sys


class GA_base:

    def __init__(self, bounds, fitnessFunction, distribution_comparison=None, populationSize=10, mutationRate=1.0,
                 percent_to_preserve=0.5,
                 selection_function=4, mutate_function=3, breed_function=2,
                 tournament_size=2, island_model=False, co_evolution=False, graph=False,
                 ui_plot=True, read_from_file=True, read_from_file_name="sim_val1"):
        print('GA BASE')

        # number of processes to be used. the higher the number the more
        # simulations can be run at once increasing the algorithms speed
        global p
        p = Pool(6)

        # GA parameters
        # print('bounds: {} : {}'.format(type(bounds), bounds))
        self.bounds = bounds
        self.fitnessFunction = fitnessFunction
        self.distribution_comparison = distribution_comparison
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.percent_to_preserve = percent_to_preserve
        #  selection function values:
        #  0 = roulette wheel (point wise gene selection)(exploitation)
        #  1 = rank (slow exploitation)
        #  2 = tournament (exploration but depends on tournament size)
        #  3 = completely random (exploration)
        #  4 = boltzmann (slowly becomes exploitation)
        self.selection_function = selection_function
        print('selection function: {}'.format(self.selection_function))
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
        self.co_evolution = co_evolution
        self.tournament_size = tournament_size

        self.current_time = time.time()

        if read_from_file:
            print('read from file')
            self.distribution_comparison.set_exp_val(
                np.asarray(self.read_from_file(read_from_file_name)))  # set before normalization

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

        self.fitness = [0] * populationSize
        self.corrected_fitness = [0] * populationSize
        self.original_fitness = [0] * populationSize
        # print(self.fitness)

        # GA breeding
        self.sumFitness = 0
        self.indexes = list()
        for i in range(populationSize):
            self.indexes.append(i)

        self.child_indexes = list()
        for i in range(int(len(self.population) * self.percent_to_preserve), populationSize):
            self.child_indexes.append(i)
        print(self.child_indexes)

        self.best = None
        self.bestScore = 0

        # simulation
        self.total_generations = 50
        self.current_generation = -1

        # visuals
        self.scale = 1
        self.graph = graph
        self.ui_plot = ui_plot

    def create_new_indiviudual(self):
        l = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 10)
        l2 = [y for y in l]
        self.population.append(l2)

    def run(self):
        # get fitness for each individual
        # print('compute fitness')
        self.current_time = time.time()

        self.current_generation += 1
        # compute fitness
        self.sumFitness = 0
        if self.current_generation is 0:
            ff = p.map(self.computeFitness, self.indexes)
            for i in range(len(ff)):
                self.fitness[i] = ff[i][0]
                self.original_fitness[i] = ff[i][0]
                self.corrected_fitness[i] = ff[i][1]
        else:
            ff = p.map(self.computeFitness, self.child_indexes)
            for i in range(len(ff)):
                self.fitness[self.child_indexes[i]] = ff[i][0]
                self.original_fitness[self.child_indexes[i]] = ff[i][0]
                self.corrected_fitness[self.child_indexes[i]] = ff[i][1]

        print('fitness: {}'.format(self.fitness))
        print('original fitness: {}'.format(self.original_fitness))
        print('corrected fitness: {}'.format(self.corrected_fitness))

        # mean_fitness = np.sum(self.fitness) / len(self.fitness)
        # print('mean: {}'.format(mean_fitness))
        # fitness_minus_mean = list()
        # for i in range(len(self.fitness)):
        #     v = (self.fitness[i] - mean_fitness)
        #     fitness_minus_mean.append(v ** 2)
        # print(fitness_minus_mean)
        # self.variance = np.sum(fitness_minus_mean) / (len(self.fitness) - 1)
        # print('variance: {}'.format(self.variance))
        # print('standard deviation: {}'.format(np.sqrt(self.variance)))

        # find best
        bestFit = min(self.fitness)
        self.bestScore = bestFit
        self.best = self.population[self.fitness.index(bestFit)]
        if self.co_evolution:
            self.best_corrected_score = self.corrected_fitness[self.fitness.index(bestFit)]
        # print('//////////////////////////////////////////////////////////////////////////////////')
        # for i in self.fitness:
        # print(i)
        # print('fitnesses: {}'.format(self.fitness))
        print('best individual: {}'.format(self.best))
        print('best fitness: {}'.format(self.bestScore))

        for i in range(len(self.fitness)):
            if np.isinf(self.fitness[i]) or self.fitness[i] > 100000:
                self.fitness[i] = 100000

        max_cost = np.max(self.fitness)
        for j in range(len(self.fitness)):
            self.fitness[j] = max_cost - self.fitness[j]

        # print('fitnesses: {}'.format(self.fitness))
        # print('original fitness: {}'.format(self.original_fitness))

        # compute sum of fitness
        self.sumFitness += np.sum(self.fitness)

        # start new population
        if self.islands_model == False:
            self.reproduction_steps()
        if not self.islands_model:
            self.fitnessFunction(self.best, plot=True)

        print('computation time: {}'.format(time.time() - self.current_time))

    def reproduction_steps(self):
        # print('new population')
        self.newPopulation = list()
        self.newFitness = list()
        self.new_original_fitness = list()

        fitness_copy = self.fitness.copy()
        for i in range(int(self.populationSize * self.percent_to_preserve)):
            maximum = np.max(fitness_copy)
            # print(maximum)
            index = self.fitness.index(maximum)
            self.newPopulation.append(self.population[index])
            self.newFitness.append(maximum)
            self.new_original_fitness.append(self.original_fitness[index])
            fitness_copy.remove(maximum)

        # print('new fitness: {}'.format(self.newFitness))
        print('new original fitness: {}'.format(self.new_original_fitness))

        # breed new selecting parents based on roulette wheel selection
        print('breed')
        if self.selection_function == 0:
            self.reproduce_roulette()
        elif self.selection_function == 1:
            self.reproduce_rank()
        elif self.selection_function == 2:
            self.reproduce_tournament(n=self.tournament_size)
        elif self.selection_function == 3:
            self.reproduce_random()
        elif self.selection_function == 4:
            self.reproduce_boltzmann()

        # mutate individuals except for previous best
        # print('mutate')
        # print('individuals')
        # skipped_best = False
        # for j in self.newPopulation:
        #     if skipped_best is False and j is self.best:
        #         skipped_best = True
        #     else:
        #         if self.mutate_function == 1:
        #             self.mutate1(j)
        #         elif self.mutate_function == 2:
        #             self.mutate2(j)
        #         elif self.mutate_function == 3:
        #             self.mutate3(j)
        #     print(j)

        for i in range(int(len(self.newPopulation) * self.percent_to_preserve)):
            # if self.newPopulation[i] in self.population:
            # print('{} in population'.format(i))
            # print(self.newPopulation[i])
            # print(self.original_fitness[self.population.index(self.newPopulation[i])])
            # print(self.bestScore)

            # matches = [x for x in range(len(self.population)) if self.newPopulation[i] is self.population[x]]
            # matched_fitnesses = [self.original_fitness[x] for x in range(len(self.original_fitness)) if x in matches]
            # minimum = np.min(matched_fitnesses)
            # self.fitness[i] = minimum
            self.fitness[i] = self.original_fitness[self.population.index(self.newPopulation[i])]

        for i in range(int(len(self.newPopulation) * self.percent_to_preserve)):
            self.original_fitness[i] = self.fitness[i]
        print(self.newPopulation[0])
        print('corrected fitnesses: {}'.format(self.fitness))
        print()
        # set newPopulation as actual population
        self.population = self.newPopulation

    def computeFitness(self, i):
        ff = list()

        f = self.fitnessFunction(self.population[i])[0]
        corrected_f = 0
        if self.co_evolution:
            current_sigma = self.distribution_comparison.sigma
            self.distribution_comparison.set_sigma([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            corrected_f = self.distribution_comparison.compute_cost()[0]
            self.distribution_comparison.set_sigma(current_sigma)

        # print('fitness: {} + individual: {} + i:{}'.format(f, self.population[i], i))

        print(self.population[i])
        ff.append(f)
        ff.append(corrected_f)
        print('ff: {}'.format(ff))
        return ff

    def read_from_file(self, file_name):
        file = open(file_name, 'r')
        text = file.read()
        text_array = text.split('/')
        to_return = list()
        for i in range(len(text_array) - 1):
            to_return.append(float(text_array[i]))
        return to_return

    def reproduce_random(self):
        # print('reproduce random')
        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
            self.newPopulation.append(self.population[rand.randint(0, len(self.population) - 1)])

        while len(self.newPopulation) < len(self.population):
            # select parents
            parent1 = self.newPopulation[rand.randint(0, len(self.newPopulation) - 1)]
            parent2 = self.newPopulation[rand.randint(0, len(self.newPopulation) - 1)]

            # breed child
            self.newPopulation.append(self.breed(parent1, parent2))

    def reproduce_boltzmann(self, alpha=0.05, start_temperature=20):
        # print('reproduce boltzmann')
        # create probabilities
        probabilities = list()
        if self.current_generation >= self.total_generations:
            self.current_generation = self.total_generations
        # boltzmann probability
        f_max = max(self.newFitness)
        k = (1 + 100 * self.current_generation / self.total_generations)
        temperature = start_temperature * (1 - alpha) ** k

        # compute probabilities
        for i in range(len(self.newFitness)):
            probabilities.append(np.exp(-1 * ((f_max - self.newFitness[i]) / temperature)))

        # print('boltzmann probabilities: {}'.format(probabilities))
        # select individuals to carry over to next generation
        sum_of_probabilities = 0
        for i in probabilities:
            sum_of_probabilities += i

        new_probabilities = probabilities
        # while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
        #     pRand = rand.uniform(0, sum_of_probabilities)
        #     sum = 0;
        #     for j in range(len(self.population)):
        #         sum += probabilities[j]
        #         if sum >= pRand:
        #             self.newPopulation.append(self.population[j])
        #             new_probabilities.append(probabilities[j])
        #             break

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
        # print('reproduce roulette')
        new_fitness = self.newFitness
        adjusted_fitness_sum = np.sum(new_fitness)

        # while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
        #
        #     pRand = rand.uniform(0, self.sumFitness)
        #
        #     sum = 0;
        #     for j in range(len(self.population)):
        #         sum += self.fitness[j]
        #         if sum >= pRand:
        #             self.newPopulation.append(self.population[j])
        #             new_fitness.append(self.fitness[j])
        #             adjusted_fitness_sum += self.fitness[j]
        #             break

        while len(self.newPopulation) < len(self.population):
            # select parents
            parent1 = None
            parent2 = None
            pRand = rand.uniform(0, adjusted_fitness_sum)

            sum = 0;
            for j in range(len(self.newPopulation)):
                sum += new_fitness[j]
                if j == len(self.newPopulation) - 1 and sum < pRand:
                    pass
                    # print('sump1: {} , pRand: {}'.format(sum, pRand))
                if sum >= pRand:
                    parent1 = self.newPopulation[j]
                    break

            pRand = rand.uniform(0, adjusted_fitness_sum)

            sum = 0;
            for j in range(len(self.newPopulation)):
                sum += new_fitness[j]
                if j == len(self.newPopulation) - 1 and sum < pRand:
                    pass
                    # print('sump2: {} , pRand: {}'.format(sum, pRand))
                if sum >= pRand:
                    parent2 = self.newPopulation[j]
                    break

            # breed child (its safe to add them right away because adjusted sum already limits search)
            self.newPopulation.append(self.breed(parent1, parent2))

    def reproduce_tournament(self, n=2):
        print('reproduce tournament: {}'.format(n))
        # while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
        #     competitors = list()
        #     for i in range(n):
        #         competitor = rand.randint(0, len(self.population) - 1)
        #         while competitor in competitors:
        #             competitor = rand.randint(0, len(self.population) - 1)
        #         competitors.append(competitor)
        #
        #     best_fitness = 0
        #     best_competitor = None
        #     for i in competitors:
        #         if best_fitness == 0:
        #             best_fitness = self.fitness[i]
        #             best_competitor = i
        #         if best_fitness < self.fitness[i]:
        #             best_fitness = self.fitness[i]
        #             best_competitor = i
        #     winner = self.population[best_competitor]
        #     self.newPopulation.append(winner)

        while len(self.newPopulation) < len(self.population):
            competitors = list()
            for i in range(n):
                competitor = rand.randint(0, len(self.newFitness) - 1)
                while competitor in competitors:
                    competitor = rand.randint(0, len(self.newFitness) - 1)
                competitors.append(competitor)

            best_fitness = 0
            best_competitor = None
            for i in competitors:
                if best_fitness == 0:
                    best_fitness = self.newFitness[i]
                    best_competitor = i
                if best_fitness < self.newFitness[i]:
                    best_fitness = self.newFitness[i]
                    best_competitor = i
            parent1 = self.population[best_competitor]

            competitors = list()
            for i in range(n):
                competitor = rand.randint(0, len(self.newFitness) - 1)
                while competitor in competitors:
                    competitor = rand.randint(0, len(self.newFitness) - 1)
                competitors.append(competitor)

            best_fitness = 0
            best_competitor = None
            for i in competitors:
                if best_fitness == 0:
                    best_fitness = self.newFitness[i]
                    best_competitor = i
                if best_fitness < self.newFitness[i]:
                    best_fitness = self.newFitness[i]
                    best_competitor = i
            parent2 = self.population[best_competitor]

            self.newPopulation.append(self.breed(parent1, parent2))

    def reproduce_rank(self):
        # rank and probabilities are in the same order. use index later to find actual individual
        print('reproduce rank')
        # fitness_copy = list()
        # for i in range(len(self.fitness)):
        #     fitness_copy.append(self.fitness[i])
        #
        # probabilities = list()
        # rank_probabilities_sum = 0
        # rank = list()
        #
        # for i in range(len(self.population)):
        #
        #     minimum = np.min(fitness_copy)
        #     print('minimum: {}'.format(minimum))
        #     rank.append(minimum)
        #     fitness_copy.remove(minimum)
        #
        # print('rank: {}'.format(rank))
        #
        # for i in range(len(rank)):
        #     probabilities.append((i + 1) / (len(rank) * (len(rank) - 1)))
        #
        # print('probabilities: {}'.format(probabilities))
        #
        # rank_probabilities_sum = np.sum(probabilities)

        new_fitness = self.newFitness
        # new_fitness.append(rank[len(rank) - 1])
        # while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
        #
        #     pRand = rand.uniform(0, rank_probabilities_sum)
        #     sum = 0;
        #     for j in range(len(rank)):
        #         sum += probabilities[j]
        #         if sum >= pRand:
        #             self.newPopulation.append(self.population[self.fitness.index(rank[j])])
        #             new_fitness.append(rank[j])
        #             break

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
            probabilities.append((i + 1) / (len(rank) * (len(rank) - 1)))

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

    def reproduce_roulette_half(self):
        # select individuals to copy and are potential parents
        adjusted_fitness_sum = 0
        new_fitness = list()
        while len(self.newPopulation) < int(len(self.population) * (self.percent_to_preserve / 2)):

            pRand = rand.uniform(0, self.sumFitness)

            sum = 0;
            for j in range(len(self.population)):
                sum += self.fitness[j]
                if sum >= pRand:
                    self.newPopulation.append(self.population[j])
                    new_fitness.append(self.fitness[j])
                    adjusted_fitness_sum += self.fitness[j]
                    break

        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):
            p = self.population[rand.randint(0, len(self.population) - 1)]
            index_of_p = self.population.index(p)
            new_fitness.append(self.fitness[index_of_p])
            adjusted_fitness_sum += self.fitness[index_of_p]
            self.newPopulation.append(p)

        while len(self.newPopulation) < len(self.population):
            # select parents
            parent1 = None
            pRand = rand.uniform(0, adjusted_fitness_sum)

            sum = 0;
            for j in range(len(self.newPopulation)):
                sum += new_fitness[j]
                if sum >= pRand:
                    parent1 = self.newPopulation[j]
                    break

            parent2 = self.newPopulation[rand.randint(0, len(self.newPopulation) - 1)]

            # breed child (its safe to add them right away because adjusted sum already limits search)
            self.newPopulation.append(self.breed(parent1, parent2))

    def breed(self, parent1, parent2):
        if self.breed_function == 1:
            return self.breed1(parent1, parent2)
        elif self.breed_function == 2:
            return self.breed2(parent1, parent2)

    # point-wise crossover
    def breed1(self, parent1, parent2):
        child = list()
        for i in range(len(parent1)):
            if rand.choice([True, False]):
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    # subsection crossover
    def breed2(self, parent1, parent2):
        child = list()
        random1 = rand.randint(0, len(parent1) - 1)
        random2 = rand.randint(0, len(parent2) - 1)
        if random1 < random2:
            child.extend(parent2[:random1])
            child.extend(parent1[random1: random2])
            child.extend(parent2[random2:])
        else:
            child.extend(parent1[:random2])
            child.extend(parent2[random2: random1])
            child.extend(parent1[random1:])
        self.mutate3(child)

        if child is parent1 or child is parent2:
            print('parent1: {}'.format(parent1))
            print('parent2: {}'.format(parent2))
            print('child: {}'.format(child))
            self.mutate3(child)
        return child

    # mutates at most one variable
    def mutate1(self, individual):
        if rand.uniform(0, 1) < self.mutationRate:
            i = rand.randint(0, len(individual) - 1)
            individual[i] = rand.uniform(self.bounds[i, 0], self.bounds[i, 1])

    # mutates a random amount of variables
    def mutate2(self, individual):
        for i in range(len(individual)):
            if rand.uniform(0, 1) < self.mutationRate:
                individual[i] = rand.uniform(self.bounds[i, 0], self.bounds[i, 1])

    # mutates often uniformly
    def mutate3(self, individual):
        for i in range(len(individual)):
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
