# Automated Quantization of Neural Networks

###### Bachelor thesis - Miroslav Šafář (xsafar23@stud.fit.vutbr.cz)

## Karolina scripts

This folder contains scripts used for running experiments on supercomputer Karolina.
*WARNING:* This script will work only on Karolina.

Before all experiments, Mobilenet was training using `mobilenet_trail_tinyimagenet.sh` script. Scripts were run using:
```shell
qsub <script_name>
```


### Experiment 1: Per-layer asymmetric quantization of weights

- `mobilenet_nsga_approx_perlayer.sh` for running NSGA-II for 24h (using approximate method for batch normalization folding)
- `mobilenet_eval_nsga_per_layer.sh` for evaluating uniform solutions for comparison (using approximate method for batch normalization folding)
- `mobilenet_eval_nsga_per_layer_accurate.sh` for evaluating uniform solutions for comparison (using more accurate method for batch normalization folding)

- `mobilenet_eval_uniform_per_layer.sh` for evaluating uniform solutions for comparison (using approximate method for batch normalization folding)
- `mobilenet_eval_uniform_per_layer_accurate.sh` for evaluating uniform solutions for comparison (using more accurate method for batch normalization folding)
