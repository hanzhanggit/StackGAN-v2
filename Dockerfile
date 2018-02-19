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


RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl 
RUN pip install torchvision 

WORKDIR /workspace
RUN chmod -R a+w /workspace

# It has to be added for StackGAN code do work
ENV PYTHONPATH /workspace

COPY requirements.txt /workspace
RUN pip install -r /workspace/requirements.txt
