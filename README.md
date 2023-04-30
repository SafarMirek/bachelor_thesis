# Automated Quantization of Neural Networks

###### Bachelor thesis - Miroslav Šafář (xsafar23)

## Environment Setup

### Create pre-trained Mobilenet model

Switch to source directory:
```shell
$ cd src
```

To create and train mobilenet model on tiny-imagenet dataset use:
```shell
$ python3 mobilenet_tinyimagenet_train.py --alpha 0.25 --save-as mobilenet_tinyimagenet_025.keras
```

If you already have weights file for model, you can create it using:
```shell
$ python3 mobilenet_tinyimagenet_create.py --alpha 0.25 --weights-path weights_025.hfd5 --destination mobilenet_tinyimagenet_025.keras
```

## Run NSGA-II

Switch to source directory:
```shell
$ cd src
```

To run NSGA-II for per-layer assymetric quantization with approximate solution for batch normalization folding use:
```shell
$ python3 run_nsga.py --generations 20 --parent-size 16 --offspring-size 16 --logs-dir <nsga_run_log_dir> --approx
```
By default it uses pre-trained mobilenet model saved as `mobilenet_tinyimagenet_025.keras`,
to use different pre-trained model, specify parameter `--base-model-path`.

Another important parameters:
- `--per-channel` use per-channel weight quantization for convolutional layers
- `--symmetric` use symmetric quantization for weights
- `--batch-size` batch size for quantization-aware training
- `--epochs` number of epochs for partial tuning of models
- `--help` to print list of all script parameters

Then evaluate the final results with full fine-tuning of the quantized models using:
```shell
python3 nsga_evaluate.py --run <nsga_run_log_dir>
```
