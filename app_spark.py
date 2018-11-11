from modules.lr import linear_regretion as lr
from modules.spark.hyperparam_search import find_best_params

if __name__ == '__main__':

    print(" [ SPARK IMP ] ")

    dataset = 'datasets/dataset.csv'

    score, mse, r2 = lr.run_sgd(dataset, find_best_params)

    print("Score: ", score)
    print("Mean squared error: ", mse)
    print("Variance score: ", r2)
