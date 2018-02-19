FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


ENV PYTHON_VERSION=2.7
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl&& \
     /opt/conda/bin/conda clean -ya 

RUN pip install torchvision 

WORKDIR /workspace
RUN chmod -R a+w /workspace

# It has to be added for StackGAN code do work
ENV PYTHONPATH /workspace

COPY requirements.txt /workspace
RUN conda install pytorch torchvision cuda90 -c pytorch
RUN conda install --yes --file /workspace/requirements.txt
RUN while read requirement; do conda install --yes $requirement; done < requirements.txt 2>error.log

