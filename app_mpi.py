from modules.lr import linear_regretion as lr
from modules.mpi.hyperparam_search import find_best_params

if __name__ == '__main__':

    print(" [ MPI IMP ] ")

    dataset = 'datasets/dataset.csv'

    score, mse, r2, hyperparam_elapsed = lr.run_sgd(dataset, find_best_params)

    print("Score: ", score)
    print("Mean squared error: ", mse)
    print("Variance score: ", r2)
    print("Tempo da busca: %s segundos." % round(hyperparam_elapsed, 3))
