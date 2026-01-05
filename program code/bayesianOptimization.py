""" bayesianOptimization.py

Bayesian optimisation of cost functions.
"""

import numpy as np
import sklearn.gaussian_process as gp
from acquisitionFunction import expectedImprovement, probabilityImprovement

""" bayesianOptimisation

    Uses Gaussian Processes to optimise the loss function `sample_loss`.

    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        costFunction: function.
            Function to be optimized.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        n_params: integer.
            Number of parameters to be optimized
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp: sklearn.gaussian_process.gpr.GaussianProcessRegressor.
            Predefined object from GaussianProcessRegressor class that represents Gaussian process model.
        alpha: double.
            Variance of the error term of the GP.
        acquisitionFunction: string.
            Choose the acquisitionFunction to optimize and retreive the next best sample from the gaussian process,
            either 'expected_improvement' or 'probability_improvement'.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        epsilon: double.
            Precision tolerance for floats.
    """


def bayesianOptimisation(n_iters, costFunction, bounds, n_params, x0=None, n_pre_samples=10,
                          gaussian_process=None, alpha=1e-5, acquisitionFunction='expected_improvement', random_search=False, epsilon=1e-7):

    if acquisitionFunction == 'expected_improvement':
        acquisition = expectedImprovement(n_params)

    elif acquisitionFunction == 'probability_improvement':
        acquisition = probabilityImprovement(n_params)

    else: print('Please input acquisitionFunction \'expected improvement\'/\'probability improvement\'')


    x_list = []
    y_list = []
    ybest_list = []
    best_score = np.inf

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(costFunction(params, plot=True))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(costFunction(params, plot=True))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP

    if gaussian_process is not None:
        model = gaussian_process #gp.GaussianProcessRegressor(**gp_params)
        if 'X_train_' in dir(gp):
            xp = model.X_train_
            yp = model.y_train_ + model.y_train_mean

    else:
        # kernel = gp.kernels.Matern(length_scale = [100,10000,10000000,1,1,1,1,1,1,1])
        # kernel = gp.kernels.Sum(gp.kernels.Matern(),gp.kernels.Matern()) 
        kernel = gp.kernels.Matern(length_scale=1.0)
        # kernel = gp.kernels.RBF(length_scale=0.1)
        # kernel = gp.kernels.DotProduct(1,1)

        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    for n in range(n_iters):
        print("Iteration", n)
        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * acquisition.sampleFunction(x_random, model, yp)
            next_sample = x_random[np.argmax(ei), :]
            
        else:
            next_sample = acquisition.optimizeAcquisition(model, yp, bounds=bounds, n_restarts=10)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.all(np.abs(next_sample - xp) <= epsilon):
            print('duplicate')
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
            

        # Sample loss for new set of parameters
        cv_score = costFunction(next_sample,plot=True)
        if cv_score < best_score:
            best_score = cv_score

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)
        ybest_list.append(best_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)
        ybest = np.array(ybest_list)
    return xp, yp, ybest, model

if __name__ == '__main__':
    pass