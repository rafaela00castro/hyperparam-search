from numpy import random

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDRegressor

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def find_best_params(X, y):
    random.seed(42)
    max_iter = random.randint(low=5, high=20, size=100) # discrete uniform distribution
    alpha = random.uniform(low=0.001, high=0.1, size=100)

    # create random grid
    param_grid = {
        'max_iter': max_iter,
        'alpha': alpha
    }

    estimator = SGDRegressor()

    best_params = __run_search(estimator, param_grid, X, y)

    # print results
    print('Best Params:', best_params)

    return best_params

def __run_search(estimator, param_grid, X, y):

    # Random search of parameters
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        scoring='neg_mean_squared_error',
        n_iter=150, cv=3, verbose=1,
        random_state=42, n_jobs=-1
    )
    # Fit the model
    search.fit(X, y)

    return search.best_params_

