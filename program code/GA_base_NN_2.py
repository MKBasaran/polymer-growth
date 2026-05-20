import numpy as np
import random as rand
from multiprocessing import Pool

from NN import NN


class GA_base_NN_2:

    def __init__(self, bounds, fitnessFunction, distribution_comparison=None, populationSize=10, mutationRate=0.10,
                 percent_to_preserve=0.5,
                 selection_function=0, mutate_function=3, breed_function=2,
                 island_model=False, co_evolution=False, graph=False,
                 ui_plot=True, read_from_file=True, read_from_file_name="sim_val1"):
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

        if read_from_file:
            print('read from file')
            self.distribution_comparison.set_exp_val(
                np.asarray(self.read_from_file(read_from_file_name)))  # set before normalization

        # GA lists

        # run fitness once
        l = np.random.uniform(bounds[:, 0], bounds[:, 1], 10)
        l2 = [y for y in l]
        if rand.uniform(0, 1) < 0.5:
            l2[len(l2) - 1] = 0.0
        else:
            l2[len(l2) - 1] = 1.0
        self.fitnessFunction(l2)
        # set input
        self.input = list()
        for i in range(len(self.distribution_comparison.exp_norm)):
            self.input.append(self.distribution_comparison.exp_norm[i])

        print(self.input)

        self.input_size = len(self.input)
        self.output_size = len(self.bounds)
        self.number_of_hidden_layers = 1
        self.hidden_layer_size = int((self.input_size + self.output_size) / 2)

        print('input')
        print(self.input)

        self.population = list()

        for i in range(populationSize):
            nn = NN(self.input_size, self.output_size, self.number_of_hidden_layers,
                    self.hidden_layer_size)
            self.population.append(nn)

        print("[{},{},{},{}]".format(len(self.distribution_comparison.exp_norm), len(bounds), 2,
                                     int((len(self.distribution_comparison.exp_norm) + len(bounds)) / 2)))

        self.fitness = [0] * populationSize
        self.corrected_fitness = [0] * populationSize
        # print(self.fitness)

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
        self.total_generations = 30
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

        self.current_generation += 1

        # compute fitness
        self.sumFitness = 0
        ff = p.map(self.computeFitness, self.indexes)
        for i in range(len(ff)):
            self.fitness[i] = ff[i][0]
            self.corrected_fitness[i] = ff[i][1]

        # find best
        bestFit = min(self.fitness)
        self.bestScore = bestFit
        self.best = self.population[self.fitness.index(bestFit)]
        if self.co_evolution:
            self.best_corrected_score = self.corrected_fitness[self.fitness.index(bestFit)]
        # print('//////////////////////////////////////////////////////////////////////////////////')
        # for i in self.fitness:
        # print(i)
        print('best individual: {}'.format(self.best))
        print('best fitness: {}'.format(bestFit))

        for i in range(len(self.fitness)):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = 1000000

        max_cost = max(self.fitness)
        for j in range(len(self.fitness)):
            self.fitness[j] = max_cost - self.fitness[j]

        # compute sum of fitness
        for j in range(len(self.fitness)):
            self.sumFitness += self.fitness[j]

        # best parameters
        print('best parameters: {}'.format(self.best))

        # start new population
        if self.islands_model == False:
            self.reproduction_steps()
        if not self.islands_model:
            self.fitnessFunction(self.best, plot=True)
        else:
            parameters = self.population[i].compute_output(self.input)
            lower_b = self.bounds[:, 0]
            upper_b = self.bounds[:, 1]

            for i in range(len(parameters)):
                if i is len(parameters) - 1:
                    if (parameters[i] < 0.5):
                        parameters[i] = 0
                    else:
                        parameters[i] = 1
                    continue
                parameters[i] = lower_b[i] + (parameters[i] * (upper_b[i] - lower_b[i]))
            self.best_parameters = parameters

    def reproduction_steps(self):
        # print('new population')
        self.newPopulation = list()
        self.newPopulation.append(self.best)

        # breed new selecting parents based on roulette wheel selection
        # print('breed')
        if self.selection_function == 0:
            self.reproduce_roulette()
        elif self.selection_function == 1:
            self.reproduce_rank()
        elif self.selection_function == 2:
            self.reproduce_tournament(n=int(len(self.population) * 0.1))
        elif self.selection_function == 3:
            self.reproduce_random()
        elif self.selection_function == 4:
            self.reproduce_boltzmann()

        # mutate individuals except for previous best
        # print('mutate')
        print('individuals')
        skipped_best = False
        for j in self.newPopulation:
            if skipped_best is False and j is self.best:
                skipped_best = True
            else:
                if self.mutate_function == 1:
                    self.mutate1(j)
                elif self.mutate_function == 2:
                    self.mutate2(j)
                elif self.mutate_function == 3:
                    self.mutate3(j)
            print(j.compute_output())

        # set newPopulation as actual population
        self.population = self.newPopulation

    def computeFitness(self, i, plot=False):
        ff = list()

        parameters = self.population[i].compute_output(self.input)
        lower_b = self.bounds[:, 0]
        upper_b = self.bounds[:, 1]

        for i in range(len(parameters)):
            if i is len(parameters) - 1:
                if (parameters[i] < 0.5):
                    parameters[i] = 0
                else:
                    parameters[i] = 1
                continue
            parameters[i] = lower_b[i] + (parameters[i] * (upper_b[i] - lower_b[i]))
        print(parameters)

        if plot:
            self.best_parameters = parameters
        f = self.fitnessFunction(parameters, plot=plot)[0]
        corrected_f = 0

        if self.co_evolution:
            current_sigma = self.distribution_comparison.sigma
            self.distribution_comparison.set_sigma([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            corrected_f = self.distribution_comparison.compute_cost()[0]
            self.corrected_fitness[i] = corrected_f
            self.distribution_comparison.set_sigma(current_sigma)

        ff.append(f)
        ff.append(corrected_f)
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

        # boltzmann probability
        f_max = max(self.fitness)
        k = (1 + 100 * self.current_generation / self.total_generations)
        temperature = start_temperature * (1 - alpha) ** k

        # compute probabilities
        for i in range(len(self.fitness)):
            probabilities.append(np.exp(-1 * ((f_max - self.fitness[i]) / temperature)))

        # print('boltzmann probabilities: {}'.format(probabilities))
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
        # print('reproduce roulette')
        adjusted_fitness_sum = 0
        new_fitness = list()
        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):

            pRand = rand.uniform(0, self.sumFitness)

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

    def reproduce_tournament(self, n=3):
        # print('reproduce tournament')
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

    def reproduce_rank(self):
        # rank and probabilities are in the same order. use index later to find actual individual
        # print('reproduce rank')
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
        while len(self.newPopulation) < int(len(self.population) * self.percent_to_preserve):

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
        child = NN(self.input_size, self.output_size, self.number_of_hidden_layers,
                   int((self.input_size + self.output_size) / 2))
        for i in range(len(parent1.weights)):
            for j in range(len(parent1.weights[i])):
                for w in range(len(parent1.weights[i][j])):
                    if rand.uniform(0, 1) > 0.5:
                        child.weights[i][j][w] = parent1.weights[i][j][w]
                    else:
                        child.weights[i][j][w] = parent2.weights[i][j][w]
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
        for i in range(len(individual.weights)):
            for j in range(len(individual.weights[i])):
                for w in range(len(individual.weights[i][j])):
                    if rand.uniform(0, 1) < self.mutationRate:
                        if rand.uniform(0, 1) < 0.8:
                            if rand.uniform(0, 1) < 0.5:
                                individual.weights[i][j][w] = individual.weights[i][j][w] + individual.weights[i][j][
                                    w] * 0.1
                                if individual.weights[i][j][w] > 1:
                                    individual.weights[i][j][w] = 1
                            else:
                                individual.weights[i][j][w] = individual.weights[i][j][w] - individual.weights[i][j][
                                    w] * 0.1
                                if individual.weights[i][j][w] < -1:
                                    individual.weights[i][j][w] = -1
                        else:
                            individual.weights[i][j][w] = rand.uniform(-1, 1)
