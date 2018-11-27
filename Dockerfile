FROM jupyter/scipy-notebook

USER root

RUN apt-get update && apt-get upgrade -y
RUN apt-get install git -y
RUN apt-get install ssh -y
RUN apt-get install libopenmpi-dev -y

USER jovyan

RUN python --version
RUN cd /home/jovyan
RUN git clone https://github.com/rapguit/hyperparam-search.git

RUN pip install mpi4py
