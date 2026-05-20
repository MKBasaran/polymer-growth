from distributionComparison import medianFoldNorm, translationInvariant
from simulation import polymer
import numpy as np
from evolutionaryAlgorithm import EvolutionaryAlgorithm

if __name__ == '__main__':

    file_format = "Data/polymer_{}.xlsx"
    datasets = ["10k", "30k", "50k", "300k"]
    all_bounds = [np.array(x) for x in [
        [[1000, 5000], [100000, 150000], [25000000, 35000000],[0, 1], [0, 0.0001], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]],
        [[4000, 7000], [100000, 150000], [30000000, 50000000],[0, 1], [0, 0.0001], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]],
        [[5000, 8000], [100000, 150000], [50000000, 100000000],[0, 1], [0, 0.0001], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]],
        [[8000, 10000], [100000, 150000], [300000000, 500000000],[0, 1], [0, 0.0001], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]]
    ]]
    experiments = [{"dataset": file_format.format(s), "bounds":b} for s,b in zip(datasets, all_bounds)]
    bests = []
    for experiment in experiments:
        file_name = experiment["dataset"]
        bounds = experiment["bounds"]
        print(file_name)
        MFC = medianFoldNorm(file_name, polymer)
        ea = EvolutionaryAlgorithm(bounds, 15, MFC.costFunction ,n=4, ui_plot=False, )
        ea.run(20)
        b = ea.get_ultimate_best()
        bests.append(b)
        print(b)

    print(bests)




