# A benchmark for random search implementations
___

In this work, a colleague and I performed a benchmark between MPI4Py and PySpark for a hyperparameter search method known as Random Search. We applied the Scikit-learn implementation as a baseline. Experimental results show that MPI implementation for random search during the training phase of 1000 regression models was almost 3x faster than Spark.

*disclaimer: This work provides only an initial analysis. For fair comparison, these experiments should also be performed for big data and more complex machine learning models.*

### Docker Build: building the image with MPI and Spark

``
$ docker build -t hsearch .
``

### Docker Run: container execution

``
$ docker run --rm -d -p 8888:8888 --name=random-hsearch hsearch
``

### Docker Exec: using to the container terminal to execute .py scripts 

``
$ docker exec -it random-hsearch /bin/bash
``

### Execution of .py scripts 

```
$ cd hyperparam-search
```

Scripts that execute the Random Search method for the problem of optimizing hyperparameters in parallel.
```
$ python app_sklearn.py

$ python app_mpi.py

$ python app_spark.py
```

Script to benchmark the implementations above. When running `python benchmark.py` by default the script will perform 10 iterations for each framework, defining a seed to generate the same random numbers, and thus ensure reproducible results. To change the number of runs and generate different hyperparameters for each run (variability in results), run the script as indicated below.
```
$ python benchmark.py -i 2 -s none

$ exit
```

### Docker Conteiner Stop: stopping and removing the container
``
$ docker container stop random-hsearch
``
