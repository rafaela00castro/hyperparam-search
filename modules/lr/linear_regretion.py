import time
from numpy import loadtxt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def load_data(train, test):
    X_train = loadtxt(train, usecols=(0), unpack=True, delimiter=',').T
    y_train = loadtxt(train, unpack=True, usecols=(1), delimiter=',')

    X_test = loadtxt(test, usecols=(0), unpack=True, delimiter=',').T
    y_test = loadtxt(test, unpack=True, usecols=(1), delimiter=',')

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1), y_test, X_val.reshape(-1, 1), y_val


def run_sgd(train, test, find_best_params):

    X_train, y_train, X_test, y_test, X_val, y_val = load_data(train, test)
    print('Data loaded!')

    hyperparam_elapsed = time.time()
    best_params = find_best_params(X_train, y_train)
    hyperparam_elapsed = time.time() - hyperparam_elapsed

    model = SGDRegressor(
        alpha=best_params['alpha'],
        max_iter=best_params['max_iter'],
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = model.score(X_val, y_val)

    pred = preds.reshape(len(preds))
    real = y_test

    mse = mean_squared_error(real, pred)
    r2 = r2_score(real, pred)

    return score, mse, r2, hyperparam_elapsed
