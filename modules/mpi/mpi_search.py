import argparse
import itertools
import random
import numpy
import settings

from mpi4py import MPI

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

from modules.lr.linear_regretion import load_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_jobs", type=int)
    parser.add_argument("-s", action="store", dest="seed")

    return parser.parse_args()


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    result = []
    random_state = get_args().seed
    settings.init(random_state)

    if rank == 0:
        numpy.random.seed(settings.seed)
        max_iter = numpy.random.randint(low=5, high=20, size=100)
        alpha = numpy.random.uniform(low=0.001, high=0.1, size=100)

        param_grid = [
            max_iter,
            alpha
        ]

        param_table = list(itertools.product(*param_grid))
        ten_percent = int(len(param_table) * 0.1)
        param_table = random.sample(param_table, ten_percent)

        train = 'datasets/test.csv'
        test = 'datasets/test.csv'
        X_train, y_train, X_test, y_test, X_val, y_val = load_data(train, test)

        data = {
            "X": X_train,
            "y": y_train,
            "X_t": X_test,
            "y_t": y_test,
            "grid": param_table
        }

    else:
        data = None

    data = comm.bcast(data, root=0)
    n_jobs = get_args().n_jobs

    chunk_len = int(len(data['grid']) / n_jobs)
    offset = chunk_len * rank

    data_chunk = numpy.asarray(data['grid'])[offset:(offset+chunk_len)]
    min_mse = 1000
    best_param = {}
    for set in data_chunk:
        model = SGDRegressor(
            alpha=set[1],
            max_iter=set[0],
            random_state=settings.seed
        )

        model.fit(data['X'], data['y'])
        preds = model.predict(data['X_t'])

        pred = preds.reshape(len(preds))
        real = data['y_t']

        mse = mean_squared_error(real, pred)

        if mse < min_mse:
            min_mse = mse
            best_param = set

    if rank == 0:
        result.append([min_mse, best_param])
        for i in range(1, n_jobs):
            p_res = comm.recv(source=i)
            result.append(p_res)

        result = min(result, key=lambda x: x[0])
        print('{},{},{}'.format(result[0], result[1][0], result[1][1]))
        MPI.Finalize()

    else:
        comm.send([min_mse, best_param], dest=0)
