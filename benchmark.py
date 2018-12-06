from modules.lr import linear_regretion as lr

from app_sklearn import run as sklearn_run
from app_mpi import run as mpi_run
from app_spark import run as spark_run

import argparse as arg
import numpy as np

def get_arguments():
	parser = arg.ArgumentParser()
	parser.add_argument('-i', '--iteractions', action='store', dest='iteractions', default=10)

	return parser.parse_args()

def execute(train, test, n_iteractions, run, description):
    print(description)
    
    scores, mses, r2s, time = [], [], [], []
    
    for i in range(n_iteractions):
        score, mse, r2, hyperparam_elapsed = run(train, test)
        scores.append(round(score, 4))
        mses.append(round(mse, 4))
        r2s.append(round(r2, 4))
        time.append(round(hyperparam_elapsed, 4))
        
    print("=======================")
    print("Mean and standard deviation after {} iteractions".format(n_iteractions))
    print("Score: {} - {}".format(np.mean(scores), np.std(scores)))
    print("Mean squared error: {} - {}".format(np.mean(mses), np.mean(mses)))
    print("Variance score: {} - {}".format(np.mean(r2s), np.std(r2s)))
    print("Tempo da busca (segundos): {} - {}".format(np.mean(time), np.std(time)))
    print("-----------------------")
    print("Lists with all the values")
    print("Score: ",scores)
    print("Mean squared error: ", mses)
    print("Variance score: ", r2s)
    print("Tempo da busca (segundos): ", time)
    print("=======================")
    

if __name__ == '__main__':

    print(" [ BENCHMARK: SKLEARN, MPI, SPARK ] ")
    
    train = 'datasets/test.csv'
    test = 'datasets/test.csv'
    
    args = get_arguments()
    
    params = [train, test, int(args.iteractions)]
    
    execute(*params, sklearn_run, " [ SKLEARN IMP ] ")
    
    execute(*params, mpi_run, " [ MPI IMP ] ")
    
    execute(*params, spark_run, " [ SPARK IMP ] ")


