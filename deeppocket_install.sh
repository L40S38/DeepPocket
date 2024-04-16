#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda create -n deeppocket -y
conda activate deeppocket

pip install molgrid
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge fpocket
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c conda-forge biopython -y
#pip install -U ProDy
conda install -c conda-forge rdkit -y
conda install -c conda-forge scikit-learn -y
conda install -c conda-forge scikit-image -y
conda install -c conda-forge wandb -y
pip install prody
pip install torch torchaudio torchvision