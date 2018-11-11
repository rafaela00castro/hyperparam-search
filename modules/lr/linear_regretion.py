import time
from numpy import loadtxt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def run_sgd(dataset, find_best_params):

    X = loadtxt(dataset, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
    Y = loadtxt(dataset, unpack=True, usecols=(11), delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print('Data loaded!')

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
