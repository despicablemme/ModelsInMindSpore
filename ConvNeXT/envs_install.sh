#!/bin/bash

YOUR_ENV_NAME=$1
YOUR_CONDA_PATH=$2

# Install gcc-7.3.0 in your env of miniconda.
conda create -n $YOUR_ENV_NAME --clone base
source ${YOUR_CONDA_PATH}/bin/activate $YOUR_ENV_NAME
conda install -c ehmoussi gcc_impl_linux-64

echo ${YOUR_CONDA_PATH}/envs/${YOUR_ENV_NAME}/libexec/gcc/x86_64-conda_cos6-linux-gnu/7.3.0/gcc
echo ${YOUR_CONDA_PATH}/envs/${YOUR_ENV_NAME}/bin/gcc

ln -s ${YOUR_CONDA_PATH}/envs/${YOUR_ENV_NAME}/libexec/gcc/x86_64-conda_cos6-linux-gnu/7.3.0/gcc ${YOUR_CONDA_PATH}/envs/${YOUR_ENV_NAME}/bin/gcc
conda install -c ehmoussi gcc_linux-64

# Install other dependencies.
conda install gmp=6.1.2 nccl openssl openmpi
