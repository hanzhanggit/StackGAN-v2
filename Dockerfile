FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

# System packages
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


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda     
ENV PATH=/opt/conda/bin:${PATH}
RUN conda update -y conda
RUN conda install --yes \
	numpy \
	pyyaml \
	scipy \
	ipython \
	pip \
	mkl \
	pytorch torchvision -c pytorch

WORKDIR /workspace
COPY requirements.txt /workspace

RUN pip install -r /workspace/requirements.txt
