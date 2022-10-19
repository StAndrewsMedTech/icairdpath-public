# docker build -t --build-arg GIT_USER=<your_user> --build-arg GIT_TOKEN=d0e4467c63... --build-arg DATA_WRITERS_GROUP_ID=<data writers group id> <your_image_name> .
FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

LABEL maintainer="iCAIRD Pathology Team"

# git arguments
ARG GIT_USER
ARG GIT_TOKEN

# permission arguments
ARG DATA_WRITERS_GROUP_ID

# update the image and install some basic software
RUN apt-get -y update
RUN apt-get -y install
RUN apt-get -y install build-essential sudo git vim curl

# set up the ubuntu user
RUN adduser ubuntu --disabled-password --gecos ''
RUN usermod -a -G sudo ubuntu

# set it so that sudo does not need a password
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# set up the data permissions
RUN groupadd -g ${DATA_WRITERS_GROUP_ID} icaird-data-writers
RUN usermod -a -G icaird-data-writers ubuntu

# change to the ubuntu user
USER ubuntu

# setup anaconda
WORKDIR "/tmp"
RUN \
    curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh \
    && bash Anaconda3-2019.10-Linux-x86_64.sh -b \
    && rm Anaconda3-2019.10-Linux-x86_64.sh
ENV PATH /home/ubuntu/anaconda3/bin:$PATH
RUN conda update conda && conda update anaconda && conda update --all

# clone the code from github into /home/ubuntu/icairdpath and change pwd to that dir
WORKDIR /home/ubuntu/
RUN git clone https://${GIT_USER}:${GIT_TOKEN}@github.com/davemor/icairdpath.git

# user make file to set up the conda environment
SHELL ["/bin/bash", "-c"]
WORKDIR /home/ubuntu/icairdpath
RUN make create_environment
RUN source activate icairdpath \
    && make requirements  \
    && pip install -e . \
    && make setup_jupyter
