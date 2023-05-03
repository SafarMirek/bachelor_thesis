#!/usr/bin/env bash
#PBS -q qgpu_exp
#PBS -N testTFEnv
#PBS -l select=1:ngpus=2,walltime=00:05:00
#PBS -A OPEN-20-37

echo "Tensorflow environment test"

cd /scratch/project/open-20-37/safarmirek/quantization/bachelor_thesis/src || exit 

echo "Install module TensorFlow/2.10.1-foss-2022a-CUDA-11.7.0"

ml TensorFlow/2.10.1-foss-2022a-CUDA-11.7.0

echo "Getting python version"
which python3
python3 --version

echo "Creating virtual environment"

python3 -m venv venv

echo "Activating venv"

source venv/bin/activate

echo "Installing required packages using pip"

python3 -m pip install -U setuptools wheel pip packaging
python3 -m pip install tensorflow==2.11.0
python3 -m pip install tensorflow-model-optimization==0.7.3
python3 -m pip install tensorflow-datasets

echo "Print environment details"
python3 tf_environment_test.py
