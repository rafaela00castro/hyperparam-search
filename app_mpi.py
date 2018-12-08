from modules.lr import linear_regretion as lr
from modules.mpi.hyperparam_search import find_best_params

import settings 

def run(train, test):
    return lr.run_sgd(train, test, find_best_params)

if __name__ == '__main__':

    print(" [ MPI IMP ] ")

    train = 'datasets/test.csv'
    test = 'datasets/test.csv'

    settings.init()
    score, mse, r2, hyperparam_elapsed = run(train, test)

    print("Score: ", score)
    print("Mean squared error: ", mse)
    print("Variance score: ", r2)
    print("Tempo da busca: %s segundos." % round(hyperparam_elapsed, 3))
