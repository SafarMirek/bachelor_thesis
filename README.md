# Automated Quantization of Neural Networks

###### Bachelor thesis - Miroslav Šafář (xsafar23@stud.fit.vutbr.cz)

## File structure

- `src/` - source code of proposed system
- `karolina/` - scripts used for running experiments on Karolina supercomputer
- `nsga_runs/` - folder that contains system runs referred to in the paper
- `latex/` - source code of the paper
- `automated_quantization_of_neural_networks_paper.pdf` - bachelor thesis paper
- `README.md` - README
- `tiny_imagenet100.tar` - dataset used for experiments
- `environment_linux_x86.yml` - conda environment file for Linux
- `environment_macos_arm64.yml` - conda environment file for macOS with Apple Silicon

## Project Setup

### Preferred: Conda

We provide conda environment file to setup conda environment.

On Apple Silicon computers:

```shell
$ conda create --file environment_macos_arm64.yml
```

On Linux:

```shell
$ conda create --file environment_linux_x86.yml
```

### PIP

There is another option to setup the project environment using pip.
We recommend using Python 3.10.9 and a virtual environment. You can install all required packages using pip:

```shell
$ pip install -r requirements_macos_arm64.txt
```

**WARNING:** You can use this option only with Apple Silicon computer. With linux please use conda environment.

### Tinyimagenet dataset

For testing purposes, we provide our tiny-imagenet dataset. To use it you need to extract 
`tiny_imagenet100.tar` into your TensorFlow Datamodels folder (default: `~/tensorflow_datasets`).

### Create pre-trained Mobilenet model

Switch to the source directory:

```shell
$ cd src
```

To create and train MobileNet model on tiny-imagenet dataset use:

```shell
$ python3 mobilenet_tinyimagenet_train.py --alpha 0.25 --save-as mobilenet_tinyimagenet_025.keras
```

If you already have a weights file for a model, you can create it using:

```shell
$ python3 mobilenet_tinyimagenet_create.py --alpha 0.25 --weights-path weights_025.hfd5 --destination mobilenet_tinyimagenet_025.keras
```

## Run NSGA-II

Switch to the source directory:

```shell
$ cd src
```

To run NSGA-II for per-layer asymmetric quantization with the approximate solution for batch normalization folding use:

```shell
$ python3 run_nsga.py --generations 20 --parent-size 16 --offspring-size 16 --logs-dir <nsga_run_log_dir> --approx
```

By default, it uses pre-trained MobileNet model saved as `mobilenet_tinyimagenet_025.keras`,
to use different pre-trained model, specify parameter `--base-model-path`.

Other important parameters:

- `--per-channel` use per-channel weight quantization for convolutional layers
- `--symmetric` use symmetric quantization for weights
- `--batch-size` batch size for quantization-aware training
- `--epochs` number of epochs for partial tuning of models
- `--multigpu` run on multiple gpus
- `--help` to print list of all script parameters

Then evaluate the final results with full fine-tuning of the quantized models using:

For evaluation of per-layer asymmetric quantization using more accurate method for batch normalization folding use:

```shell
python3 nsga_evaluate.py --run <nsga_run_log_dir>
```

For evaluation of per-layer asymmetric quantization using approximate method for batch normalization folding use:

```shell
python3 nsga_evaluate.py --run <nsga_run_log_dir> --approx
```

For evaluation of per-channel symmetric quantization use:

```shell
python3 nsga_evaluate.py --run <nsga_run_log_dir> --per-channel --symmetric
```

Other important parameters:

- `--batch-size` batch size for quantization-aware training
- `--epochs` number of epochs for final fine-tuning
- `--multigpu` run on multiple gpus
- `--help` to print a list of all script parameters

## Visualization of results

To view system results use:
```shell
python3 show_layer_configuration.py --run <nsga_run_log_dir> [--per-channel] [--symmetric]
```
This script allows you to choose between best-found configurations and then shows you the bit-width for each layer.

