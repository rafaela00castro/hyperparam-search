import os


def find_best_params(X, y):

    best_params = __run_search()

    # print results
    print('Best Params:', best_params)

    return best_params

def __run_search():
    max_iter = None
    alpha = None

    print('Exec parallel job')
    exec_result = os.popen("mpiexec -n 4 python modules/mpi/mpi_search.py 4").read()
    print('done!')

    print('Result:')
    print(exec_result)

    result = exec_result.split(',')

    return {
        'max_iter': int(float(result[1])),
        'alpha': float(result[2])
    }

