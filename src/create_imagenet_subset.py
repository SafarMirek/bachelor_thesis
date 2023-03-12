# Author: ANDREW TU
# From https://www.kaggle.com/code/tusonggao/create-imagenet-train-subset-100k
import os
import time
import random

train_path = '/Users/miroslavsafar/kaggle/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train'
new_train_path = '/Users/miroslavsafar/kaggle/working/imagenet-subset-300/train'

os.system(f'rm -rf {new_train_path}')
os.system(f'mkdir -p {new_train_path}')

filenames = []

dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
print('len(dirs): ', len(dirs))


def get_all_files(directory):
    filenames = []
    class_name = directory.split('/')[-1]
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
            part_name = filename.split('_')[0]
            assert part_name == class_name
    return filenames


process_start_t = time.time()
part_start_t = time.time()

classes = len(dirs)

number_per_class = [320 for _ in range(classes)]
number_of_classes_with_one_more = 291
classes_with_one_more = random.sample([i for i in range(classes)], k=number_of_classes_with_one_more)
for i in classes_with_one_more:
    number_per_class[i] = number_per_class[i] + 1

for i, directory in enumerate(dirs):
    directory_path = train_path + '/' + directory
    filenames = get_all_files(directory_path)
    filenames = random.sample(filenames, number_per_class[i])
    assert len(filenames) == number_per_class[i]

    os.system(f"mkdir {new_train_path + '/' + directory}")

    for filename in filenames:
        src_file_path = train_path + '/' + directory + '/' + filename
        tgt_file_path = new_train_path + '/' + directory + '/' + filename
        os.system(f'cp {src_file_path} {tgt_file_path}')

    if (i + 1) % 100 == 0:
        print(f'directory: {directory} filenames: {filenames[:3]}')
        print(f'now i: {i + 1}, cost time: {time.time() - part_start_t:.2f} sec')
        part_start_t = time.time()

print(f'finished, total cost time: {time.time() - process_start_t:.2f} sec')
