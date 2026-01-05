import numpy as np
from simulation import polymer
from hillClimbing import hillClimbing
from distributionComparison import minMaxNorm, medianFoldNorm, translationInvariant
from evolutionaryAlgorithm import EvolutionaryAlgorithm
from bayesianOptimization import bayesianOptimisation
from helper import processArguments
import pickle



if __name__ == '__main__':
    # Compare distributions after normalizing by either minMaxNorm or medianFoldNorm.
    # compareDist =  minMaxNorm('Data/polymer_30k.xlsx', polymer)
    #compareDist = medianFoldNorm('Data/polymer_5k.xlsx', polymer)
    compareDist = translationInvariant('Data/polymer_300k.xlsx', polymer)

    param_boundaries = np.array([[8000, 10000], [90000, 100000], [300000000, 400000000],
                                 [0, 1], [0, 0.0001], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]])
    # X0 = np.array([[1000, 100000, 31600000, 0.2, 0.0000806, 0.5, 0.67, 0.67, 1, 1]])

    # Use compareDistribution object
    # hillClimbing(compareDist.costFunction, processArguments)

    # alg = evolutionaryAlgorithm(param_boundaries, 10, compareDist.costFunction, processArguments)
    # print(alg.run(100))
    # print(alg.population)

    # model = pickle.load(open("save.p", "rb"))
    xp, yp, ybest, model = bayesianOptimisation(n_iters=150, costFunction=compareDist.costFunction, bounds=param_boundaries,
                                         n_params=10,
                                         x0=None, n_pre_samples=1, gaussian_process=None, alpha=0.1,
                                         acquisitionFunction='expected_improvement', epsilon=1e-5)
    # np.savetxt(fname='testitest', X=yp)
    np.save(file='BO_pre_10_MAT_4_5k', arr=ybest)
    pickle.dump(model, open("BO_pre_10_MAT_4_5k.p", "wb"))


