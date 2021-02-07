from modules.lr import linear_regretion as lr

from app_sklearn import run as sklearn_run
from app_mpi import run as mpi_run
from app_spark import run as spark_run

import argparse as arg
import numpy as np
import settings 

def get_arguments():
    parser = arg.ArgumentParser()
    parser.add_argument('-i', '--iterations', action='store', dest='iterations', default=10)
    parser.add_argument('-s', '--seed', action='store', dest='seed', default=42)

    return parser.parse_args()

def execute(train, test, n_iterations, run, description):
    print(description)
    
    scores, mses, r2s, time = [], [], [], []
    
    for i in range(n_iterations):
        score, mse, r2, hyperparam_elapsed = run(train, test)
        scores.append(round(score, 5))
        mses.append(round(mse, 5))
        r2s.append(round(r2, 5))
        time.append(round(hyperparam_elapsed, 5))
        
    print("\n=======================")
    print("Mean and standard deviation after {} iterations".format(n_iterations))
    print("Score: media:{0:.5f}, std: {1:.5f}".format(np.mean(scores), np.std(scores)))
    print("Mean squared error: media: {0:.5f}, std: {1:.5f}".format(np.mean(mses), np.std(mses)))
    print("Variance score: media: {0:.5f}, std: {1:.5f}".format(np.mean(r2s), np.std(r2s)))
    print("Tempo da busca (segundos): media: {0:.5f}, std: {1:.5f}".format(np.mean(time), np.std(time)))
    print("-----------------------")
    print("Lists with all the values")
    print("Score: ", scores)
    print("Mean squared error: ", mses)
    print("Variance score: ", r2s)
    print("Tempo da busca (segundos): ", time)
    print("=======================\n")
    

if __name__ == '__main__':

    print(" [ BENCHMARK: SKLEARN, MPI, SPARK ] ")
    
    train = 'datasets/test.csv'
    test = 'datasets/test.csv'
    
    args = get_arguments()
    settings.init(args.seed)
    
    params = [train, test, int(args.iterations)]
    
    execute(*params, sklearn_run, " [ SKLEARN IMP ] ")
    
    execute(*params, mpi_run, " [ MPI IMP ] ")
    
    execute(*params, spark_run, " [ SPARK IMP ] ")


