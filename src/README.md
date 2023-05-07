# Automated Quantization of Neural Networks

###### Bachelor thesis - Miroslav Šafář (xsafar23@stud.fit.vutbr.cz)

## Source code structure

Packages:

- `tf_quantize/` - contains source code for TensorFlow support of per-layer and mixed-precision quantization
- `nsga/` - contains source code of NSGA-II implementation used in the proposed system
- `visualize/` - contains common methods used in out visualization scripts
- `datasets/` - contains generated package for tiny_imagenet100 creation and scripts for accessing tiny-imagenet and
  cifar-10 datasets

Important scripts:

- `mobilenet_tinyimagenet_create.py` - creates keras model of Mobilenet in H5 format from weights file
- `mobilenet_tinyimagenet_qat.py` - this scripts and also library used in out QAT NSGA-II implementation runs
  quantization-aware training for given model and tiny-imagenet dataset
- `mobilenet_tinyimagenet_train.py` - trains floating-point model of mobilenet with tiny-imagenet dataset

- `run_nsga.py` - runs NSGA-II with mobilenet and tiny-imagenet dataset
- `nsga_evaluate.py` - runs final evaluation on specified run of NSGA-II
- `show_layer_configuration.py` - show per-layer bit-width assigment of NSGA-II results



