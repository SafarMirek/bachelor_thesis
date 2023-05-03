#!/usr/bin/env bash
#PBS -q qgpu
#PBS -N qatNSGAApprox
#PBS -l select=1:ngpus=8,walltime=24:00:00
#PBS -A OPEN-20-37

echo "Mobilenet QAT NSGA Approx with disabled act quantization"

# Change to the local scratch directory of the job
cd /lscratch/$PBS_JOBID || exit

echo "Copy src to local scratch"

cp -vrf /home/${USER}/bachelor_thesis/src/* .

echo "Copy datasets to local scratch for better performance"

cp -vrf /home/${USER}/tensorflow_datasets/ .
export TFDS_DATA_DIR=/lscratch/$PBS_JOBID/tensorflow_datasets

echo "Create symlinks to nsga cache and runs"

ln -s /mnt/proj2/open-20-37/safarmirek/nsga_runs nsga_runs
ln -s /mnt/proj2/open-20-37/safarmirek/nsga_cache cache

echo "Local files"
ls -l

echo "Install module TensorFlow/2.10.1-foss-2022a-CUDA-11.7.0"

ml TensorFlow/2.10.1-foss-2022a-CUDA-11.7.0
 
echo "Creating virtual environment"

python3 -m venv venv

echo "source venv/bin/activate"

source venv/bin/activate

echo "Installing required packages using pip"

python3 -m pip install -U setuptools wheel pip packaging
python3 -m pip install tensorflow==2.11.0
python3 -m pip install tensorflow-model-optimization==0.7.3
python3 -m pip install protobuf==3.19.6 tensorflow-datasets==4.8.2
python3 -m pip install py-paretoarchive
python3 -m pip install protobuf==3.19.6

echo "Running Mobilenet QAT NSGA Approx with disabled act quantization and per-layer asymmetric weight quantization"
python3 run_nsga.py --multigpu --learning-rate 0.0025 --batch-size 64 --approx --act-quant-wait 12 --qat-epochs 12 --generations 50 --logs-dir nsga_runs/mobilenet_025_qat_12_no_act_approx_per_layer_asymmetric_24pch --parent-size 24 --offspring-size 24 --cache-datasets --previous-run nsga_runs/mobilenet_025_qat_12_no_act_approx_per_layer_asymmetric_24pch
