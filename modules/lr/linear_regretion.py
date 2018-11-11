import time
from numpy import loadtxt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def run_sgd(train, test, find_best_params):

    X_train = loadtxt(train, usecols=(0), unpack=True, delimiter=',').T
    y_train = loadtxt(train, unpack=True, usecols=(1), delimiter=',')

    X_test = loadtxt(test, usecols=(0), unpack=True, delimiter=',').T
    y_test = loadtxt(test, unpack=True, usecols=(1), delimiter=',')

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print('Data loaded!')
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    X_val = X_val.reshape(-1, 1)

    hyperparam_elapsed = time.time()
    best_params = find_best_params(X_train, y_train)
    hyperparam_elapsed = time.time() - hyperparam_elapsed

    model = SGDRegressor(
        learning_rate=best_params['learning_rate'],
        eta0=best_params['eta0'],
        alpha=best_params['alpha'],
        max_iter=700,
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
