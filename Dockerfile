FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

WORKDIR /transformer

ENV LANG C.UTF-8

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    wget \
    vim

# Install python
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O Miniconda.sh && \
    bash Miniconda.sh -b -p /root/miniconda3 && \
    rm Miniconda.sh
ENV PATH /root/miniconda3/bin:$PATH

# Cache heavy libraries first
RUN pip install --no-cache-dir numpy==1.16.5
RUN pip install --no-cache-dir torch==1.7.0

# Install requirements
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

COPY . .
