"""imagenet150 dataset."""
import io
import logging
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.datasets.imagenet2012 import imagenet_common


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for imagenet150 dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain imagenet150_train.tar file
    """

    def _info(self) -> tfds.core.DatasetInfo:
        names_file = imagenet_common.label_names_file()
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(encoding_format='jpeg'),
                'label': tfds.features.ClassLabel(names_file=names_file),
                'file_name': tfds.features.Text(),  # Eg: 'n15075141_54.JPEG'
            }),
            supervised_keys=('image', 'label'),
            homepage='https://image-net.org/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        train_path = os.path.join(dl_manager.manual_dir, 'imagenet150_train.tar')
        splits = []
        _add_split(
            split_list=splits,
            split=tfds.Split.TRAIN,
            split_path=train_path,
            dl_manager=dl_manager,
        )

        # TODO(imagenet150): Returns the Dict[split names, Iterator[Key, Example]]
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
