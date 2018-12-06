from modules.lr import linear_regretion as lr
from modules.sklearn.hyperparam_search import find_best_params

def run(train, test):
    return lr.run_sgd(train, test, find_best_params)

if __name__ == '__main__':

    print(" [ SKLEARN IMP ] ")

    train = 'datasets/test.csv'
    test = 'datasets/test.csv'

    score, mse, r2, hyperparam_elapsed = run(train, test)

    print("Score: ", score)
    print("Mean squared error: ", mse)
    print("Variance score: ", r2)
    print("Tempo da busca: %s segundos." % round(hyperparam_elapsed, 3))