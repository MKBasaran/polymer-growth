import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

""" 
	Acquisition function.

	Arguments:
	----------
		x: array-like, shape = [n_samples, n_hyperparams]
			The point for which the expected improvement needs to be computed.
		gaussian_process: GaussianProcessRegressor object.
			Gaussian process trained on previously evaluated hyperparameters.
		evaluated_loss: Numpy array.
			Numpy array that contains the values off the loss function for the previously
			evaluated hyperparameters.
		greater_is_better: Boolean.
			Boolean flag that indicates whether the loss function is to be maximised or minimised.
		n_params: int.
			Dimension of the hyperparameter space.
"""


class acquisitionFunction:
	def __init__(self, n_params, greater_is_better=False):
		self.greater_is_better = greater_is_better
		self.n_params = n_params

	def sampleFunction(self):
		pass

	""" 
		optimizeAcquisition

		Proposes the next hyperparameter to sample the loss function for.

		Arguments:
		----------
			acquisition_func: function.
				Acquisition function to optimise.
			gaussian_process: GaussianProcessRegressor object.
				Gaussian process trained on previously evaluated hyperparameters.
			evaluated_loss: array-like, shape = [n_obs,]
				Numpy array that contains the values off the loss function for the previously
				evaluated hyperparameters.
			greater_is_better: Boolean.
				Boolean flag that indicates whether the loss function is to be maximised or minimised.
			bounds: Tuple.
				Bounds for the L-BFGS optimiser.
			n_restarts: integer.
				Number of times to run the minimiser with different starting points.
	"""

	def optimizeAcquisition(self, gp, previous_costs, bounds=(0, 10), n_restarts=25):
		best_x = None
		best_acquisition_value = 1


		for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, self.n_params)):

			res = minimize(fun=self.sampleFunction,
						   x0=starting_point.reshape(1, -1),
						   bounds=bounds,
						   method='L-BFGS-B',
                           # method = 'BFGS',
						   args=(gp, previous_costs))

			if res.fun < best_acquisition_value:
				best_acquisition_value = res.fun
				best_x = res.x

		return best_x


class expectedImprovement(acquisitionFunction):
	def __init__(self, n_params):
		super().__init__(n_params)


	def sampleFunction(self, x_predict, gp, previous_costs):
		x_predict = x_predict.reshape(-1, self.n_params)
		mu, sigma = gp.predict(x_predict, return_std=True)

		if self.greater_is_better:
			optimum_cost = np.max(previous_costs)
		else:
			optimum_cost = np.min(previous_costs)

		scaling_factor = (-1) ** (not self.greater_is_better)

		# In case sigma equals zero
		with np.errstate(divide='ignore'):
			Z = scaling_factor * (mu - optimum_cost-0.01) / sigma
			expected_improvement = scaling_factor * (mu - optimum_cost-0.01) * norm.cdf(Z) + sigma * norm.pdf(Z)
			expected_improvement[sigma == 0.0] == 0.0
		return -1 * expected_improvement


class probabilityImprovement(acquisitionFunction):
	def __init__(self, n_params):
		super().__init__(n_params)

	def sampleFunction(self, x_predict, gp, previous_costs):

		eps=.01
		x_predict = x_predict.reshape(-1, self.n_params)

		mu, sigma = gp.predict(x_predict, return_std=True)

		if self.greater_is_better:
			loss_optimum = np.max(previous_costs)
		else:
			loss_optimum = np.min(previous_costs)

		scaling_factor = (-1) ** (not self.greater_is_better)


		with np.errstate(divide='ignore'):
			Z = scaling_factor * (mu - loss_optimum - 0.01) / sigma
			probability_improvement = norm.cdf(Z)
			probability_improvement[sigma == 0.0] == 0.0

		return -1 * probability_improvement
