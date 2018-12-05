# hyperparam-search
___

Computação paralela e distribuída

Exercício de disciplpina.

### Docker Build: construção da imagem com MPI e Spark

``
$ docker build -t hsearch .
``

### Docker Run: execução do container

``
$ docker run --rm -d -p 8888:8888 --name=random-hsearch hsearch
``

### Docker Exec: chamada ao terminal do container para executar scripts .py 

``
$ docker exec -it random-hsearch /bin/bash
``

### Execução dos script .py 

```
$ cd hyperparam-search

$ python app_sklearn.py

$ python app_mpi.py

$ python app_spark.py

exit
```

### Docker Conteiner Stop: parada e remoção do container

``
$ docker container stop random-hsearch
``
