#!/usr/bin/env bash
#PBS -q qgpu
#PBS -N mnQuantW
#PBS -l select=1:ngpus=1,walltime=04:00:00
#PBS -A OPEN-20-37

cd /scratch/project/open-20-37/safarmirek/quantization/bachelor_thesis/src || exit 

ml TensorFlow
 
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -U setuptools wheel pip
python3 -m pip install tensorflow==2.11.0
python3 -m pip install tensorflow-model-optimization==0.7.3

echo "MobileNet Quantize 8bit Weights"
CUDA_VISIBLE_DEVICES=0 python3 mobilenet_quantize_v2.py --warmup 0.05 -v --cache -e 50
