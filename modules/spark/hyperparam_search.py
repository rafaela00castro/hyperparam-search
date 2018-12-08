from pyspark.sql import SparkSession
from pyspark.mllib.random import RandomRDDs
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import numpy as np
import settings

def find_best_params(X, y):
    spark, sc = __start_session()
    
    param_size = 100
    sample_size = 500
    parallelism = sc.defaultParallelism
    
    train_numpy = np.concatenate((X, y[:, np.newaxis]), axis=1)
    train = sc.parallelize(train_numpy) \
              .map(lambda r: [Vectors.dense(r[0]),float(r[1])]) \
              .toDF(['features','label']) \
              .repartition(parallelism) \
              .cache()
    
    reg_param = RandomRDDs.uniformRDD(sc, size=param_size, seed=settings.seed) \
                          .map(lambda x: 0.001 + (0.1 - 0.001) * x) \
                          .collect()

    max_iter = RandomRDDs.uniformRDD(sc, size=param_size, seed=settings.seed) \
                          .map(lambda x: int(5 + (20 - 5) * x)) \
                          .collect()

    # create random grid
    estimator = LinearRegression(solver='normal')
    param_grid = ParamGridBuilder().addGrid(estimator.regParam, reg_param) \
                                   .addGrid(estimator.maxIter, max_iter) \
                                   .build()

    param_grid = sc.parallelize(param_grid) \
                    .takeSample(withReplacement=False, num=sample_size, seed=settings.seed)

    best_params = __run_search(estimator, param_grid, train, parallelism)
    
    train.unpersist()
    spark.stop()

    # print results
    print('Best Params:', best_params)

    return best_params


def __run_search(estimator, param_grid, train, parallelism):
    kfold = 2
    grid_size = len(param_grid)
    eval = RegressionEvaluator(metricName='mse')
        
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=param_grid, 
                        evaluator=eval, numFolds=kfold,
                        seed=settings.seed, parallelism=parallelism)

    print('Fitting {} folds for each of {} candidates, totalling {} fits'
          .format(kfold, grid_size, kfold * grid_size))
    model = cv.fit(train)

    # check the best parameters
    best_model = model.bestModel
    best_params = best_model.extractParamMap()
    max_iter_key = best_model.getParam('maxIter')
    reg_param_key = best_model.getParam('regParam')

    return {
        'max_iter': best_params[max_iter_key],
        'alpha': best_params[reg_param_key]
    }

def __start_session():
    spark = SparkSession.builder \
                        .master('local[*]') \
                        .appName('Random-search') \
                        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')
    print('Versão do Spark: ', sc.version)
    
    return spark, sc