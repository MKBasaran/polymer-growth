import numpy as np

from GA_base import GA_base
import random as rand

from co_evolution import sigma_GA


class island_GA:
    def __init__(self, bounds, corrected_bounds, fitnessFunction, fitnessfunction2, distribution_comp,
                 populationSize=10, mutationRate=1,
                 migration_number=5, number_of_islands=2, graph=False, ui_plot=True, read_from_file=False,
                 read_from_file_name='exp_val0'):
        # GA parameters
        self.bounds = bounds
        self.corrected_bounds = corrected_bounds
        self.distribution_comp = distribution_comp
        self.fitnessFunction = fitnessFunction
        self.fitnessFunction2 = fitnessfunction2
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.migration_number = migration_number
        if migration_number * number_of_islands >= populationSize - 1:
            self.migration_number = int(populationSize / number_of_islands - 2)
        if self.migration_number < 0:
            self.migration_number = 0
        self.generation = -1
        self.number_of_islands = number_of_islands
        self.read_from_file_name = read_from_file_name

        # load file from fake data
        if read_from_file:
            print('read from file')
            self.distribution_comp.set_exp_val(
                np.asarray(self.read_from_file(self.read_from_file_name)))  # set before normalization

        # find original sigma but no longer important for island model
        self.original_sigma = list()
        for i in range(len(distribution_comp.sigma)):
            self.original_sigma.append(distribution_comp.sigma[i])

        # start GA's
        print('islands: {}'.format(number_of_islands))

        # make copys of the distribution comparison, a.k.a., cost function
        # so the islands wont interfere with each other
        self.distribution_comp_list = list()
        for i in range(number_of_islands):
            self.distribution_comp_list.append(self.distribution_comp.copy())
        self.algorithms = list()

        # initialize the islands by creating basic GA's
        for i in range(number_of_islands):
            if i == 0:
                print('i: {} value: 0'.format(i))
                self.algorithms.append(
                    GA_base(bounds, self.distribution_comp_list[i].costFunction,
                            self.distribution_comp_list[i], populationSize,
                            selection_function=2, tournament_size=8, island_model=True, read_from_file=read_from_file,
                            read_from_file_name=read_from_file_name))

            elif i == 1:
                print('i: {} value: 1'.format(i))
                self.algorithms.append(
                    GA_base(bounds, self.distribution_comp_list[i].costFunction,
                            self.distribution_comp_list[i], populationSize,
                            selection_function=0, island_model=True, read_from_file=read_from_file,
                            read_from_file_name=read_from_file_name))
            elif i == 2:
                print('i: {} value: 2'.format(i))
                self.algorithms.append(
                    GA_base(bounds, self.distribution_comp_list[i].costFunction,
                            self.distribution_comp_list[i], populationSize,
                            selection_function=1, island_model=True, read_from_file=read_from_file,
                            read_from_file_name=read_from_file_name))
            elif i == 3:
                print('i: {} value: 3'.format(i))
                self.algorithms.append(
                    GA_base(bounds, self.distribution_comp_list[i].costFunction,
                            self.distribution_comp_list[i], populationSize,
                            selection_function=2, island_model=True, read_from_file=read_from_file,
                            read_from_file_name=read_from_file_name))

        # creates sigmas but are no longer important
        self.sigma_GA_list = list()
        for i in range(number_of_islands):
            self.sigma_GA_list.append(
                sigma_GA(original_sigma=self.original_sigma,
                         distribution_comparison=self.distribution_comp_list[i]))

        self.best = None

        # visuals
        self.scale = 1
        self.graph = graph
        self.ui_plot = ui_plot

    def run(self):
        self.generation += 1
        # let each island perform a run
        for a in self.algorithms:
            a.run()
            print()

        # once every X generation migration takes place.
        if self.generation % 7 == 0:
            self.migrate_circle()

        # let the islands reproduce
        for i in self.algorithms:
            i.reproduction_steps()

        # find best individual from the islands
        b = None
        s = 10000000000
        index = 0
        for i in range(len(self.algorithms)):
            current_score = self.algorithms[i].bestScore
            if current_score < s:
                s = current_score
                b = self.algorithms[i].best
                index = i

        self.best_score = s

        # print results
        print()
        print('RESULTS')
        print('best algorithm: {}'.format(index))
        print('global best fitness: {}'.format(s))
        for i in self.sigma_GA_list:
            print(i.best)
        print()

        self.best = b

        self.fitnessFunction(self.best, plot=True)

    def read_from_file(self, file_name):
        file = open(file_name, 'r')
        text = file.read()
        text_array = text.split('/')
        to_return = list()
        for i in range(len(text_array) - 1):
            to_return.append(float(text_array[i]))
        return to_return

    # fully connected migration function but i highly suggest you use circle migration
    def migrate_fully_connected(self):
        print('migrate fully connected')
        if self.number_of_islands < 2:
            return
        # for now assume that communication topology is fully connected
        populations = list()
        for i in self.algorithms:
            populations.append(i.population)

        # find which individuals to migrate for each island
        population_migrating = dict()
        for i in range(len(self.algorithms)):
            for j in range(len(self.algorithms)):
                if i == j:
                    continue
                individuals = list()
                while len(individuals) < self.migration_number:
                    r = rand.randint(0, len(self.algorithms[i].population) - 1)
                    if self.algorithms[i].population[r] not in individuals:
                        if self.algorithms[i].breed_function == 0 and self.algorithms[i].population[r] == \
                                self.algorithms[i].best:
                            pass
                        else:
                            individuals.append(self.algorithms[i].population[r])

                population_migrating['{}{}'.format(i, j)] = individuals

        # migrate selected individuals to all islands
        for i in range(len(self.algorithms)):
            for j in range(len(self.algorithms)):
                if i == j:
                    continue
                self.algorithms[j].population.extend(population_migrating['{}{}'.format(i, j)])

        # remove worst individuals from each island
        for i in range(len(self.algorithms)):
            for j in range(len(self.algorithms)):
                if i == j:
                    continue
                for k in range(len(population_migrating['{}{}'.format(i, j)])):
                    if population_migrating['{}{}'.format(i, j)][k] in self.algorithms[i].population:
                        self.algorithms[i].population.remove(population_migrating['{}{}'.format(i, j)][k])

    # circle migration
    def migrate_circle(self):
        print('migrate string')
        # check if migration is possible
        if self.number_of_islands < 2:
            return

        # migrate from bottom island to top
        for i in range(len(self.algorithms), 0, -1):

            # if in combination with the if several lines below creates the connection between
            # the first and last in the line of islands. removing this would turn this migration into string
            if i is len(self.algorithms):
                i = 0

            # find best individuals to migrate
            to_migrate = list()
            pop = self.algorithms[i].population
            current_fitness = self.algorithms[i].original_fitness

            for j in range(self.migration_number):
                minimum = None
                for k in range(len(current_fitness)):
                    if minimum is None and current_fitness[k] not in to_migrate and current_fitness[k] not in self.algorithms[i - 1].original_fitness:
                        minimum = current_fitness[k]
                    elif minimum is not None:
                        if minimum > current_fitness[k] and current_fitness[k] not in to_migrate and current_fitness[k] not in self.algorithms[i - 1].original_fitness:
                            minimum = current_fitness[k]
                if minimum is not None:
                    to_migrate.append(minimum)

            # second part of connecting first and last island
            if i is 0:
                i = len(self.algorithms)

            print('to migrate: {}'.format(to_migrate))

            # add fitness from selected individuals to next island in line
            self.algorithms[i - 1].original_fitness.extend(to_migrate)

            # find the individuals of the corresponding fitnesses
            for j in range(len(to_migrate)):
                to_migrate[j] = pop[current_fitness.index(to_migrate[j])]

            # add selected individuals to the next island
            self.algorithms[i - 1].population.extend(to_migrate)

            # remove worst individuals from next island
            to_remove = list()
            current_fitness = self.algorithms[i - 1].original_fitness
            for j in range(len(to_migrate)):
                maximum = None
                for k in range(len(current_fitness)):
                    if maximum is None and current_fitness[k] not in to_remove:
                        maximum = current_fitness[k]
                    elif maximum is not None:
                        if maximum < current_fitness[k] and current_fitness[k] not in to_remove:
                            maximum = current_fitness[k]
                to_remove.append(maximum)

            # remove worst fitnesses from next island
            for j in range(len(to_remove)):
                index = self.algorithms[i - 1].original_fitness.index(to_remove[j])
                self.algorithms[i - 1].original_fitness.pop(index)
                self.algorithms[i - 1].population.pop(index)

            # transform the fitnesses to be used since the previously mentioned fitnesses are actually costs
            max_original = np.max(self.algorithms[i - 1].original_fitness)
            for j in range(len(self.algorithms[i - 1].original_fitness)):
                self.algorithms[i - 1].fitness[j] = max_original - self.algorithms[i - 1].original_fitness[j]

            print('size: {} alg: {}'.format(len(self.algorithms[i - 1].original_fitness), i - 1))
            print('current fitness')
            print(self.algorithms[i - 1].original_fitness)

    # star migration
    def migrate_star(self):
        print('migrate star')
        # check if migration is possible
        if self.number_of_islands < 2:
            return

        # find individuals to migrate to the central island
        to_migrate_fitness = list()
        to_migrate_population = list()
        for i in range(self.number_of_islands - 1, 0, -1):
            current_fitness = self.algorithms[i].fitness
            for j in range(self.migration_number):
                maximum = None
                for k in range(len(current_fitness)):
                    if maximum is None and current_fitness[k] not in to_migrate_fitness:
                        maximum = current_fitness[k]
                    elif maximum is not None:
                        if maximum < current_fitness[k] and current_fitness[k] not in to_migrate_fitness:
                            maximum = current_fitness[k]
                to_migrate_fitness.append(maximum)
                to_migrate_population.append(self.algorithms[i].population[current_fitness.index(maximum)])


        # add the selected individuals to the central island
        self.algorithms[0].population.extend(to_migrate_population)
        self.algorithms[0].fitness.extend(to_migrate_fitness)


        # remove worst individuals from central island
        to_remove_fitness = list()
        to_remove_population = list()
        for j in range(len(to_migrate_population)):
            lowest = None
            for k in range(len(self.algorithms[0].fitness)):
                if lowest is None and self.algorithms[0].fitness[k] not in to_remove_fitness:
                    lowest = self.algorithms[0].fitness[k]
                elif lowest is not None:
                    if lowest > self.algorithms[0].fitness[k] and self.algorithms[0].fitness[
                        k] not in to_remove_fitness:
                        lowest = self.algorithms[0].fitness[k]
            to_remove_fitness.append(lowest)
            to_remove_population.append(self.algorithms[0].population[self.algorithms[0].fitness.index(lowest)])

        for i in to_remove_fitness:
            self.algorithms[0].fitness.remove(i)
        for i in to_remove_population:
            self.algorithms[0].population.remove(i)

        for i in self.algorithms:
            print('pop {} - fit {}'.format(len(i.population), len(i.fitness)))
