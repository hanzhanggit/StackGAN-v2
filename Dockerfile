# Start from an NVIDIA Docker file to have GPU acces already setup
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Arian Hosseini <arian.hosseini@elementai.com>

ARG user_id
ARG user_name
ARG work_dir

# Install some packages I need
RUN apt-get update && apt-get install -y \
    ttf-freefont \
    vim \
    nano \
    python-pip \
    ssh \
    git \
    python-tk \
    eog \
    tmux \
    ipython

# Upgrade pip
RUN pip install --upgrade pip

# Install pytorch for pyton 2.7
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl

#http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
RUN pip install torchvision

# Install some python dependences I need in my project
RUN pip install --upgrade pip
RUN pip install \
    matplotlib \
    plotly \
    dominate \
    future \
    scipy numpy six cycler pytz subprocess32 python-dateutil backports.functools-lru-cache pyparsing olefile Pillow decorator networkx PyWavelets scikit-image pandas seaborn \
    nltk tensorboardX fuel easydict torchfile sklearn tqdm

RUN pip install flask
# Add your own user to do not use root user
RUN adduser --uid ${user_id} ${user_name} 

# Set the working directory to /eai, and set the HOME environment
# variable too.
WORKDIR /eai/project
ENV HOME /eai/project

CMD ["/bin/bash"]
