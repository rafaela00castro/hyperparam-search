from numpy import random

from sklearn.linear_model import SGDRegressor


def find_best_params(X, y):
    eta0 = random.uniform(low=0.00001, high=0.001, size=100)
    alpha = random.uniform(low=0.001, high=0.1, size=100)

    # create random grid
    param_grid = {
        'learning_rate': ['optimal', 'invscaling'],
        'eta0': eta0,
        'alpha': alpha
    }

    estimator = SGDRegressor()

    best_params = __run_search(estimator, param_grid, X, y)

    # print results
    print('Best Params:', best_params)

    return best_params


def __run_search(estimator, param_grid, X, y):
    lr = None
    eta0 = None
    alpha = None

    # TODO implementar o metodo aqui para selecionar os mellhores param

    return {
        'learning_rate': lr,
        'eta0': eta0,
        'alpha': alpha
    }

