# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@fit.vutbr.cz)
# This script was inspired by ANDREW TU from https://www.kaggle.com/code/tusonggao/create-imagenet-train-subset-100k

import argparse
import os
import random
from xml.dom.minidom import parse
from os import path


def ensure_clear_dir(dir_path: str):
    """
    Ensures directory is clear and exists
    :param dir_path: Path to the directory
    """
    os.system(f'rm -rf {dir_path}')
    os.system(f'mkdir -p {dir_path}')


def get_all_files_in_directory(directory_path: str):
    """
    Returns all files in directory
    :param directory_path: Path to the directory
    :return: List of found files
    """
    filenames = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            filenames.append(filename)
    return filenames


def main(*, train_path: str, train_path_dest: str, validation_path_data: str, validation_path_annotations: str,
         validation_path_dest: str, number_of_classes: int, number_per_class: int):
    """

    Creates a mobilenet subset from original mobilenet data, you can specify number of classes
    and number of samples per each class

    Validation data are converted from format annotated using xml files to directory annotations
    that can be better processed by TensorFlow Datasets library

    :param train_path: Path to training data directory
    :param train_path_dest: Path to destination training data
    :param validation_path_data: Path to validation data directory
    :param validation_path_annotations: Path to validation annotations directory
    :param validation_path_dest: Path to destination validation data
    :param number_of_classes: Number of classes of new subset
    :param number_per_class: Number of samples per class
    """
    ensure_clear_dir(train_path_dest)
    ensure_clear_dir(validation_path_dest)

    dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    num_of_available_classes = len(dirs)

    if number_of_classes > num_of_available_classes:
        raise ValueError("There are less classes than required")

    # Randomly choose required number of classes to process
    class_dirs = random.sample(dirs, k=number_of_classes)

    for class_directory in class_dirs:
        directory_path = os.path.join(train_path, class_directory)
        filenames = get_all_files_in_directory(directory_path)

        # Randomly choose required number of samples (all if the number of samples is smaller than required)
        filenames = random.sample(filenames, min(number_per_class, len(filenames)))
        if len(filenames) < number_of_classes:
            # Print warning that the class does not have required number of samples
            print(f"WARN: {class_directory} has less than {number_of_classes} images")
        if len(filenames) == 0:
            continue

        # Create directory for class in destination directory
        os.system(f"mkdir {os.path.join(train_path_dest, class_directory)}")
        os.system(f"mkdir {os.path.join(validation_path_dest, class_directory)}")

        for filename in filenames:
            src_file_path = train_path + '/' + class_directory + '/' + filename
            tgt_file_path = train_path_dest + '/' + class_directory + '/' + filename
            os.system(f'cp {src_file_path} {tgt_file_path}')

    for val_filename in get_all_files_in_directory(validation_path_data):
        val_basename = path.basename(val_filename)
        val_annotation_filename = val_basename.replace(".JPEG", ".xml")
        with open(validation_path_annotations + "/" + val_annotation_filename, "r") as annotation_file:
            annotation = parse(annotation_file)
            root = annotation.documentElement
            names = root.getElementsByTagName("name")
            label = str(names[0].firstChild.nodeValue)
            if label in dirs:
                print("Class for " + val_filename + " is " + label)
                src_file_path = validation_path_data + '/' + val_filename
                tgt_file_path = validation_path_dest + '/' + label + '/' + val_filename
                os.system(f'cp {src_file_path} {tgt_file_path}')

    print("Creating of subset was successful.")


if __name__ == "__main__":
    # Script arguments
    parser = argparse.ArgumentParser(
        prog='create_imagenet_subset',
        description='Creates subset of imagenet',
        epilog='')

    parser.add_argument("--train-path",
                        default="imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train")

    parser.add_argument("--train-path-dest", default="datasets_temp/tiny-imagenet100/train")

    parser.add_argument("--validation-path-data",
                        default="imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val")

    parser.add_argument("--validation-path-annotations",
                        default="imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val")

    parser.add_argument("--validation-path-dest", default="datasets_temp/tiny-imagenet100/val")

    parser.add_argument("--classes", default=100, type=int)
    parser.add_argument("--per-class", default=1000, type=int)

    args = parser.parse_args()

    main(train_path=args.train_path, train_path_dest=args.train_path_dest,
         validation_path_data=args.validation_path_data, validation_path_annotations=args.validation_path_annotations,
         validation_path_dest=args.validation_path_dest,
         number_of_classes=args.classes,
         number_per_class=args.per_class
         )
