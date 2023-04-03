# Author: ANDREW TU, Edited by Miroslav Safar
# From https://www.kaggle.com/code/tusonggao/create-imagenet-train-subset-100k
import os
import time
import random
from xml.dom.minidom import parse
from os import path

train_path = '/Users/miroslavsafar/kaggle/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train'
new_train_path = '/Users/miroslavsafar/kaggle/working/tiny-imagenet100/train'

validation_path_data = '/Users/miroslavsafar/kaggle/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
validation_path_annotations = '/Users/miroslavsafar/kaggle/imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val'
new_validation_path = '/Users/miroslavsafar/kaggle/working/tiny-imagenet100/val'

os.system(f'rm -rf {new_train_path}')
os.system(f'mkdir -p {new_train_path}')

os.system(f'rm -rf {new_validation_path}')
os.system(f'mkdir -p {new_validation_path}')

filenames = []

dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
print('len(dirs): ', len(dirs))


def get_all_files(directory):
    filenames = []
    class_name = directory.split('/')[-1]
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
    return filenames


process_start_t = time.time()
part_start_t = time.time()

number_of_classes = 100
number_per_class = 1000

classes = len(dirs)

if number_of_classes > classes:
    raise ValueError("There are less classes than required")

dirs = random.sample(dirs, k=number_of_classes)
print('new len(dirs): ', len(dirs))

for i, directory in enumerate(dirs):
    directory_path = train_path + '/' + directory
    filenames = get_all_files(directory_path)
    filenames = random.sample(filenames, min(number_per_class, len(filenames)))
    if len(filenames) < number_of_classes:
        print(f"{directory} has less than {number_of_classes} images")
    if len(filenames) == 0:
        continue

    os.system(f"mkdir {new_train_path + '/' + directory}")
    os.system(f"mkdir {new_validation_path + '/' + directory}")

    for filename in filenames:
        src_file_path = train_path + '/' + directory + '/' + filename
        tgt_file_path = new_train_path + '/' + directory + '/' + filename
        os.system(f'cp {src_file_path} {tgt_file_path}')

    if (i + 1) % 100 == 0:
        print(f'directory: {directory} filenames: {filenames[:3]}')
        print(f'now i: {i + 1}, cost time: {time.time() - part_start_t:.2f} sec')
        part_start_t = time.time()

for val_filename in get_all_files(validation_path_data):
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
            tgt_file_path = new_validation_path + '/' + label + '/' + val_filename
            os.system(f'cp {src_file_path} {tgt_file_path}')

print(f'finished, total cost time: {time.time() - process_start_t:.2f} sec')
