import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

""" distributionComparison
    Parent class for any combination of normalization and cost function. 
    Also does some preprocessing to load an experimental dataset.
    

    Arguments:
    ----------
        file_name: string.
        Define location of .xls file containing experimental dataset
        simulation: function.
        Pointer towards simulation function returning
        a simulated distribution of polymer lengths.
        fig: boolean.
        Plot a comparison between the experimental- and simulated distribution
        of polymer lengths.
"""


class distributionComparison:
    def __init__(self, file_name, simulation, fig=None):

        self.file_counter = 0
        self.sim = simulation
        self.exp_df = pd.read_excel(file_name)
        self.exp_molmass = self.exp_df[
            self.exp_df.columns[0]].values  # First column of the dataframe corresponding to the molar_mass
        self.exp_values = self.exp_df[self.exp_df.columns[1]].values  # Values for  linear differential molar mass
        self.exp_chainlen = ((self.exp_molmass - 180) / 99.13).astype(int)  # Convert molar_mass to polymer lenghts
        self.cl_max = self.exp_chainlen[-1]
        self.exp_cl_val = dict(zip(
            np.arange(self.cl_max),
            np.zeros(self.cl_max)))  # Dictionary with polymer lengths initialzed with zeros
        for cl in self.exp_cl_val:  # Fill each polymer lengths with a linear differential molar mass
            ind = np.where(cl == self.exp_chainlen)
            if not len(ind[0]) == 0:
                self.exp_cl_val[cl] = self.exp_values[ind[0][0]]

        for cl in self.exp_cl_val.keys():
            if (self.exp_cl_val[cl] == 0) & (cl != 0):
                self.exp_cl_val[cl] = self.exp_cl_val[cl - 1]

        self.exp_val = np.array(list(self.exp_cl_val.values()))

        if not fig:  # Initialize plot figure
            self.fig = plt.figure(figsize=(15, 5))
        else:
            self.fig = fig
        self.ax0, self.ax1, self.ax2 = self.fig.add_subplot(131), self.fig.add_subplot(132), self.fig.add_subplot(133)
        self.lowest_cost = np.inf  # Set initial best value to infinity

    def preprocessDist(self, dead, living, coupled, save_file=False):  # Preprocess distribution from simulation output
        print('preprocessing')
        sim_data = np.concatenate((dead, living, coupled))
        sim_cl_max = sim_data.max()
        sim_val, sim_bins = np.histogram(sim_data, bins=np.arange(sim_cl_max + 1))

        diff = int(sim_cl_max - self.exp_val.shape[0])

        # write simulation to file
        if save_file:
            print('sim_val{}'.format(self.file_counter))
            file = open('exp_val{}'.format(self.file_counter), 'w')
            for i in self.exp_val:
                file.write('{}/'.format(str(i)))
            file.close()
        self.file_counter += 1

        if diff > 0:
            exp_val = np.concatenate((self.exp_val, np.zeros(abs(diff))))
        else:
            sim_val = np.concatenate((sim_val, np.zeros(abs(diff))))
            exp_val = self.exp_val

        return exp_val, sim_val

    def costFunction(self, arguments, plot=False):
        pass

    # Input arguments to generate simulation results and return
    # difference with experimental data

    def plotDistributions(self, exp_norm, sim_norm, cost):
        print('exp_norm max: {}'.format(np.max(exp_norm)))
        print('sim norm max: {}'.format(np.max(sim_norm)))
        self.ax0.clear()
        self.ax0.bar(np.arange(exp_norm.shape[0]), exp_norm)
        self.ax0.set_title('Experiment')
        self.ax1.clear()
        self.ax1.bar(np.arange(sim_norm.shape[0]), sim_norm)
        self.ax1.set_title('Simulation')
        if cost < self.lowest_cost:
            self.lowest_cost = cost
            self.ax2.clear()
            self.ax2.bar(np.arange(sim_norm.shape[0]), sim_norm)
            self.ax2.set_title('Best Match')
        plt.pause(1e-40)


""" minMaxNorm
    First implemented normalization and cost function

"""


class minMaxNorm(distributionComparison):
    def __init__(self, file_name, simulation, fig=None):
        super().__init__(file_name, simulation, fig)

    def costFunction(self, arguments, plot=False):

        dead, living, coupled = self.sim(*arguments)  # Run simulation polymer growth
        exp_val, sim_val = self.preprocessDist(dead, living, coupled)

        # Normalize both exp- and sim-data by min-max normalization
        exp_val_max = exp_val.max()
        exp_norm = exp_val / exp_val_max
        exp_norm_sum = np.sum(exp_norm)

        sim_val_max = sim_val.max()
        sim_norm = sim_val / sim_val_max
        sim_norm_sum = np.sum(sim_norm)

        # Compute difference by l2-norm
        if exp_norm_sum > sim_norm_sum:
            cost = np.sum(abs(exp_norm - sim_norm)) / (sim_norm_sum / exp_norm_sum) ** 2

        else:
            cost = np.sum(abs(exp_norm - sim_norm)) / (exp_norm_sum / sim_norm_sum) ** 2

        if plot:
            # print(arguments)
            self.plotDistributions(exp_norm, sim_norm, cost)

        return cost


""" medianFoldNorm
    Second implemented normalization and cost function

"""


class medianFoldNorm(distributionComparison):
    def __init__(self, file_name, simulation, sigma=None, fig=None):
        super().__init__(file_name, simulation, fig)
        self.median_foldNorm = 1  # Initial median_foldNorm
        if sigma is None:
            self.sigma = [1, 5, 5, 5, 5, 5]  # Initial weights for each part of distribution
        else:
            self.sigma = sigma

    def costFunction(self, arguments, plot=False):

        dead, living, coupled = self.sim(*arguments)
        exp_val, sim_val = self.preprocessDist(dead, living, coupled)

        foldNorm = np.divide(exp_val, sim_val, out=np.zeros(sim_val.shape),
                             where=sim_val != 0)  # Division by 0 returns 0
        median_foldNorm = np.median(foldNorm[foldNorm.nonzero()])  # Compute median from all folds, excluding zero's
        if not np.isfinite(
                median_foldNorm): median_foldNorm = self.median_foldNorm  # If no valid fold use previous foldNorm
        self.median_foldNorm = median_foldNorm

        # NOTE: comment median_foldnorm to not normalize
        sim_norm = sim_val * median_foldNorm  # Normalize simulation values by medianFoldNorm

        exp_norm = exp_val

        # Cost function based on weights by standard deviation
        exp_sd, exp_mean = np.std(exp_norm), np.mean(exp_norm)
        cost = 0
        # Weight each part of the distribution
        for i in range(len(self.sigma)):
            indices = np.where((exp_norm > exp_mean - exp_sd * i) & (exp_norm < exp_mean + exp_sd * i))
            cost += np.sum(abs((exp_norm[indices] - sim_norm[indices])) ** (1 / self.sigma[i]))

        if plot:
            # print(arguments)
            self.plotDistributions(exp_norm, sim_norm, cost)
        return cost


""" translationInvariant
    Using above version of medianFoldNormalization with a different cost function.

"""


class translationInvariant(distributionComparison):
    def __init__(self, file_name, simulation, sigma=None, transfac=1, fig=None):
        super().__init__(file_name, simulation, fig)
        self.median_foldNorm = 1  # Initial medianFoldNorm
        if sigma is None:
            self.sigma = [1, 1, 1, 1, 1, 1]  # Initial weights for each part of distribution
        else:
            self.sigma = sigma
        self.transfac = transfac

    def set_sigma(self, new_sigma):
        self.sigma = new_sigma

    def costFunction(self, arguments, plot=False):

        dead, living, coupled = self.sim(*arguments)
        exp_val, sim_val = self.preprocessDist(dead, living, coupled)

        posmaxsim = np.where(sim_val == sim_val.max())
        posmaxexp = np.where(exp_val == exp_val.max())
        if len(posmaxsim[0]) > 1:
            posmaxsimvalue = posmaxsim[0][0]
        else:
            posmaxsimvalue = posmaxsim[0]
        if len(posmaxexp[0]) > 1:
            posmaxexpvalue = posmaxexp[0][0]
        else:
            posmaxexpvalue = posmaxexp[0]
        percentage = abs((posmaxsimvalue / posmaxexpvalue) - 1)  # measure relative distance of the peaks
        if len(percentage) > 1:
            print('posmaxsim: {}'.format(posmaxsim))
            print('posmaxexp: {}'.format(posmaxexp))
            print('percentage: {}'.format(percentage))
            sys.exit(0)

        f = posmaxsim[0] - posmaxexp[
            0]  # when negative move simulation data to the right. when positive move to the left

        if f[0] >= 0:  # move simulation data to the left
            cutted_sim_val = sim_val[f[0]:]
            trans_sim_val = np.append(cutted_sim_val, np.zeros(f[0]))
        if f[0] < 0:  # move simulation data to the right
            cutted_sim_val = sim_val[:len(sim_val) + f[0]]
            trans_sim_val = np.append(np.zeros(abs(f[0])), cutted_sim_val)
        print(f[0])

        foldNorm = np.divide(exp_val, trans_sim_val, out=np.zeros(trans_sim_val.shape),
                             where=trans_sim_val != 0)  # Division by 0 returns 0
        median_foldNorm = np.median(foldNorm[foldNorm.nonzero()])  # Compute median from all folds, excluding zero's
        if not np.isfinite(
                median_foldNorm): median_foldNorm = self.median_foldNorm  # If no valid fold use previous foldNorm
        self.median_foldNorm = median_foldNorm

        # NOTE: comment median foldnorm to remove normalization
        trans_sim_norm = trans_sim_val * median_foldNorm  # Normalize trans_inv simulation values by medianFoldNorm
        # sim_norm = sim_val * median_foldNorm

        exp_norm = exp_val

        # Cost function based on weights by standard deviation
        exp_sd, exp_mean = np.std(exp_norm), np.mean(exp_norm)
        cost = 0

        # original
        non_zero_indices = np.where(exp_norm > 0)
        non_zero_sim_indices = np.where(trans_sim_norm > 0)
        non_zero_sim_filtered_indices = [x for x in non_zero_sim_indices[0] if x not in non_zero_indices[0]]

        partition = int(len(non_zero_indices[0]) / len(self.sigma))
        for i in range(len(self.sigma)):
            # sigma values need to be between 0 then 1 and lower values giving a higher weight
            # due to the fact that base values are also less then one
            # original
            # indices = np.where((exp_norm > exp_mean - exp_sd * i) & (exp_norm < exp_mean + exp_sd * i))
            # cost += np.sum(abs((exp_norm[indices] - trans_sim_norm[indices])) ** (1 / self.sigma[i]))

            # new (covers everything and actually gives subsections different weights)
            partition_indices = non_zero_indices[0][i * partition: (i + 1) * partition]
            partition_cost = np.sum(
                abs((exp_norm[partition_indices] - trans_sim_norm[partition_indices])) ** (
                        1 / self.sigma[i]))  # or  * self.sigma[i]
            cost += partition_cost
            print('i: {} + partition_cost: {}'.format(i, partition_cost))

        # cost of indices outside target area
        extra_cost = 0
        for i in non_zero_sim_filtered_indices:
            extra_cost += abs(exp_norm[i] - trans_sim_norm[i])
        cost += extra_cost
        print('extra cost: {}'.format(extra_cost))

        # original
        if self.f[0] > 1000:
            cost = 10000
            print('DISTANCE LIMIT REACHED')
        else:
            cost = cost * np.exp(percentage / self.transfac)

        # new carry different weight for not being in the correct location
        # cost = cost + (percentage * self.transfac)

        print(percentage)
        if plot:
            # print(arguments)
            self.plotDistributions(exp_norm, trans_sim_norm, cost)

        return cost


# cost function used for thesis
class min_maxV2(distributionComparison):
    def __init__(self, file_name, simulation, sigma=None, transfac=1, fig=None):
        super().__init__(file_name, simulation, fig)
        self.file_name = file_name
        self.simulation = simulation
        self.fig = fig
        self.median_foldNorm = 1  # Initial medianFoldNorm
        if sigma is None:
            self.sigma = [1, 1, 1, 1, 1, 1]  # Initial weights for each part of distribution
        else:
            self.sigma = sigma
        self.transfac = transfac

    def set_sigma(self, new_sigma):
        self.sigma = new_sigma

    def set_exp_val(self, actual_exp_val):
        print('set actual exp val')
        print('original: {}'.format(self.exp_val))
        self.exp_val = actual_exp_val
        print('new: {}'.format(self.exp_val))

    def costFunction(self, arguments, plot=False, save_file=False):

        dead, living, coupled = self.sim(*arguments)  # Run simulation polymer growth
        exp_val, sim_val = self.preprocessDist(dead, living, coupled, save_file)

        if save_file:
            file = open('exp_val{}'.format(self.file_counter - 1), 'a')
            file.write(str(arguments))
            file.close()

        # Normalize both exp- and sim-data by min-max normalization
        exp_val_max = exp_val.max()
        self.exp_norm = exp_val / exp_val_max

        sim_val_max = sim_val.max()
        self.sim_norm = sim_val / sim_val_max

        # shift simulation to the correct location
        posmaxsim = np.where(self.sim_norm == self.sim_norm.max())
        posmaxexp = np.where(self.exp_norm == self.exp_norm.max())
        if len(posmaxsim[0]) > 1:
            posmaxsimvalue = posmaxsim[0][0]
        else:
            posmaxsimvalue = posmaxsim[0]
        if len(posmaxexp[0]) > 1:
            posmaxexpvalue = posmaxexp[0][0]
        else:
            posmaxexpvalue = posmaxexp[0]
        self.percentage = abs((posmaxsimvalue / posmaxexpvalue) - 1)  # measure relative distance of the peaks
        if len(self.percentage) > 1:
            print('posmaxsim: {}'.format(posmaxsim))
            print('posmaxexp: {}'.format(posmaxexp))
            print('percentage: {}'.format(self.percentage))
            sys.exit(0)

        self.f = posmaxsimvalue - posmaxexpvalue  # when negative move simulation data to the right. when positive move to the left

        if self.f[0] >= 0:  # move simulation data to the left
            cutted_sim_val = self.sim_norm[self.f[0]:]
            self.trans_sim_norm = np.append(cutted_sim_val, np.zeros(self.f[0]))
        if self.f[0] < 0:  # move simulation data to the right
            cutted_sim_val = self.sim_norm[:len(self.sim_norm) + self.f[0]]
            self.trans_sim_norm = np.append(np.zeros(abs(self.f[0])), cutted_sim_val)
        print("distance: {}".format(self.f[0]))
        print('percentage: {}'.format(self.percentage))

        # compute cost
        print('sigma: {}'.format(self.sigma))
        cost = self.compute_cost()

        if plot:
            self.plotDistributions(self.exp_norm, self.trans_sim_norm, cost)

        return cost

    def compute_cost(self):
        cost = 0
        # find non zero indices of target data
        self.non_zero_indices = np.where(self.exp_norm > 0)
        # find non zero indices of simulated data
        non_zero_sim_indices = np.where(self.trans_sim_norm > 0)
        # find non zero indices of simulated data - target data
        non_zero_sim_filtered_indices = [x for x in non_zero_sim_indices[0] if x not in self.non_zero_indices[0]]

        # create bins
        partition = int(len(self.non_zero_indices[0]) / len(self.sigma))

        # compute cost per bin
        for i in range(len(self.sigma)):
            partition_indices = self.non_zero_indices[0][i * partition: (i + 1) * partition]

            error_sum = 0.0
            for p in partition_indices:
                error = abs(self.exp_norm[p] - self.trans_sim_norm[p]) / max(self.exp_norm[p], self.trans_sim_norm[p])
                error_sum += error

            # multiply error by bin cost
            partition_cost = error_sum * self.sigma[i]

            cost += partition_cost

        # in case bins do not cover all data points
        missed_values = len(self.non_zero_indices[0]) - partition * len(self.sigma)
        partition_indices = self.non_zero_indices[0][
                            len(self.non_zero_indices[0]) - 1 - missed_values: len(self.non_zero_indices[0])]
        cost_plus = 0
        for p in partition_indices:
            cost_plus += abs(self.exp_norm[p] - self.trans_sim_norm[p]) / max(self.exp_norm[p], self.trans_sim_norm[p])
        cost += cost_plus
        # print('missed indices: {}'.format(partition_indices))
        print('cost_plus: {}'.format(cost_plus))

        # cost of indices outside target non zero indices
        extra_cost = 0
        for i in non_zero_sim_filtered_indices:
            extra_cost += abs(self.exp_norm[i] - self.trans_sim_norm[i])
        cost += extra_cost
        print('extra cost: {}'.format(extra_cost))

        # penalty for not having the peak in the correct location
        cost = cost * np.exp(self.percentage / self.transfac)

        # preventing inf values.
        if np.isinf(cost) or cost > 100000:
            cost = [100000]

        print("cost min max: {}".format(cost))
        print()
        return cost

    def copy(self):
        return min_maxV2(self.file_name, self.simulation, self.sigma, self.transfac, fig=self.fig)


class basicCostFunction(distributionComparison):
    def __init__(self, file_name, simulation, sigma=None, transfac=1, fig=None):
        super().__init__(file_name, simulation, fig)
        self.median_foldNorm = 1  # Initial medianFoldNorm
        if sigma is None:
            self.sigma = [1, 1, 1, 1, 1, 1]  # Initial weights for each part of distribution
        else:
            self.sigma = sigma
        self.transfac = transfac

    def costFunction(self, arguments, plot=False):

        dead, living, coupled = self.sim(*arguments)  # Run simulation polymer growth
        exp_val, sim_val = self.preprocessDist(dead, living, coupled)

        # Normalize both exp- and sim-data by min-max normalization
        exp_val_max = exp_val.max()
        exp_norm = exp_val / exp_val_max

        sim_val_max = sim_val.max()
        sim_norm = sim_val / sim_val_max

        cost = 0;
        for i in range(len(exp_val)):
            cost += np.abs(exp_val[i] - sim_val[i]) ** 2

        if plot:
            # print(arguments)
            self.plotDistributions(exp_norm, sim_norm, cost)

        return cost


class locationCostFunction(distributionComparison):
    def __init__(self, file_name, simulation, fig=None):
        super().__init__(file_name, simulation, fig)

    def costFunction(self, arguments, plot=False):
        dead, living, coupled = self.sim(*arguments)  # Run simulation polymer growth
        exp_val, sim_val = self.preprocessDist(dead, living, coupled)

        # Normalize both exp- and sim-data by min-max normalization
        exp_val_max = exp_val.max()
        exp_norm = exp_val / exp_val_max

        sim_val_max = sim_val.max()
        sim_norm = sim_val / sim_val_max

        max_exp = max(exp_val)
        i1 = 0
        for i in range(len(exp_val)):
            if (exp_val.data[i] == max_exp):
                i1 = i
                break
        max_sim = max(sim_val)
        i2 = 0
        for i in range(len(sim_val)):
            if (sim_val.data[i] == max_sim):
                i2 = i
                break

        cost = np.abs(i1 - i2)

        if plot:
            # print(arguments)
            self.plotDistributions(exp_norm, sim_norm, cost)

        return cost
