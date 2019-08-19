FROM continuumio/miniconda3
WORKDIR /tmp
COPY environment.yml .
RUN conda env create -f environment.yml
