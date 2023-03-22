"""tiny_imagenet100 dataset."""
import os

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.datasets.imagenet2012 import imagenet_common


def get_labels():
    return ['n02319095', 'n02094258', 'n01774750', 'n04311174', 'n03404251', 'n04286575', 'n03124170', 'n03942813',
            'n02910353', 'n02859443', 'n07579787', 'n01630670', 'n02268853', 'n02403003', 'n03594734',
            'n04366367', 'n03188531', 'n04418357', 'n03793489', 'n02113186', 'n01491361', 'n01768244', 'n02087394',
            'n04456115', 'n04501370', 'n01592084', 'n02823428', 'n02114548', 'n03888257', 'n02398521', 'n03902125',
            'n02092002', 'n03594945', 'n01944390', 'n02110185', 'n02113023', 'n03777568', 'n02879718', 'n02489166',
            'n02802426', 'n07717556', 'n07615774', 'n07734744', 'n02510455', 'n09229709', 'n02013706', 'n02096177',
            'n02823750', 'n04326547', 'n04476259', 'n04118538', 'n03532672', 'n02892767', 'n01871265', 'n02457408',
            'n04065272', 'n03240683', 'n03063599', 'n02094433', 'n02817516', 'n02120079', 'n03814639', 'n04009552',
            'n02097047', 'n01694178', 'n02091831', 'n01914609', 'n04090263', 'n02094114', 'n02104365', 'n02096294',
            'n02927161', 'n03291819', 'n03047690', 'n02206856', 'n01753488', 'n02017213', 'n01798484', 'n04522168',
            'n03187595', 'n04153751', 'n02096437', 'n03197337', 'n02980441', 'n01484850', 'n02974003', 'n01945685',
            'n04429376', 'n02276258', 'n12057211', 'n02791124', 'n01774384', 'n04493381', 'n03216828', 'n01601694',
            'n02098286', 'n04263257', 'n02111129', 'n01739381', 'n03325584']


class Builder(tfds.core.GeneratorBasedBuilder):
    """
    DatasetBuilder for tiny_imagenet100 dataset.
    Dataset contains 100 classes of 1000 images for training
    and original validation set for there classes from imagetnet
    """

    VERSION = tfds.core.Version('1.0.1')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        '1.0.1': 'Fix labels',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain tiny_imagenet100_train.tar and tiny_imagenet100_val.tar
    """

    def _info(self) -> tfds.core.DatasetInfo:
        names = get_labels()
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(encoding_format='jpeg'),
                'label': tfds.features.ClassLabel(names=names),
                'file_name': tfds.features.Text(),
            }),
            supervised_keys=('image', 'label'),
            homepage='https://image-net.org/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        train_path = os.path.join(dl_manager.manual_dir, 'tiny_imagenet100_train.tar')
        validation_path = os.path.join(dl_manager.manual_dir, 'tiny_imagenet100_val.tar')
        splits = []
        _add_split(
            split_list=splits,
            split=tfds.Split.TRAIN,
            split_path=train_path,
            dl_manager=dl_manager,
        )
        _add_split(
            split_list=splits,
            split="validation",
            split_path=validation_path,
            dl_manager=dl_manager,
        )

        return splits

    def _generate_examples(
            self, archive, validation_labels=None, labels_exist=True
    ):
        """Yields examples."""
        if not labels_exist:  # Test split
            for key, example in imagenet_common.generate_examples_test(archive):
                yield key, example
        if validation_labels:  # Validation split
            for key, example in imagenet_common.generate_examples_validation(
                    archive, validation_labels
            ):
                yield key, example
        # Training split. Main archive contains folders names after a synset noun.
        # Each folder contains pictures associated to that synset.
        for fname, image in archive:
            label = fname.split("/")[0]  # fname is something like 'n01632458/n15075141_54.JPEG'
            image_fname = fname.split("/")[1]
            # image = self._fix_image(image_fname, image)
            record = {
                'file_name': image_fname,
                'image': image,
                'label': label,
            }
            yield fname, record


def _add_split(split_list, split, split_path, dl_manager, **kwargs):
    if not tf.io.gfile.exists(split_path):
        raise FileNotFoundError(f"Archive for {split} was not found ({split_path})")
    else:
        split_list.append(
            tfds.core.SplitGenerator(
                name=split,
                gen_kwargs={
                    'archive': dl_manager.iter_archive(split_path),
                    **kwargs,
                },
            ),
        )
