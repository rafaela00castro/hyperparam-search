# hyperparam-search
___

Computação paralela e distribuída

Exercício de disciplina.

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

### Execução dos scripts .py 

``
$ cd hyperparam-search
``
Scripts que executam o método Random Search para o problema de otimização de hiperparâmetros de forma paralela.
```
$ python app_sklearn.py

$ python app_mpi.py

$ python app_spark.py
```

Script para executar o benchmark das implementações acima. Ao executar `python benchmark.py` por padrão o script executará 10 iterações para cada *framework*, definindo um *seed* para gerar os mesmos números aleatórios, e assim garantir resultados reprodutíveis. Para mudar o número de execuções e gerar diferentes hiperparâmetros a cada execução (variabilidade nos resultados), executar o script conforme indicado abaixo.

```
$ python benchmark.py -i 2 -s none

$ exit
```

### Docker Conteiner Stop: parada e remoção do container

``
$ docker container stop random-hsearch
``
