import glob
import gzip
import json
import os
import random

import numpy as np
import tensorflow as tf

import calculate_model_size
import mobilenet_tinyimagenet_qat
from nsga.nsga import NSGA
from nsga.nsga import NSGAAnalyzer
from tf_quantization.quantize_model import quantize_model
from tf_quantization.transforms.quantize_transforms import PerLayerQuantizeModelTransformer


class QATNSGA(NSGA):

    def __init__(self, logs_dir, base_model, parent_size=50, offspring_size=50, generations=25, batch_size=128,
                 qat_epochs=10, previous_run=None, cache_datasets=False, approx=False, activation_quant_wait=0,
                 per_channel=True, symmetric=True):
        super().__init__(logs_dir=logs_dir,
                         parent_size=parent_size, offspring_size=offspring_size, generations=generations,
                         objectives=[("accuracy", True), ("memory", False)], previous_run=previous_run
                         )
        self.base_model = base_model
        self.batch_size = batch_size
        self.qat_epochs = qat_epochs
        self.cache_datasets = cache_datasets
        self.approx = approx
        self.activation_quant_wait = activation_quant_wait
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.quantizable_layers = self.get_analyzer().get_quantizable_layers()

    def get_maximal(self):
        return {
            "accuracy": 1.0,
            "memory": calculate_model_size.calculate_weights_mobilenet_size(self.base_model)
        }

    def init_analyzer(self) -> NSGAAnalyzer:
        return QATAnalyzer(self.base_model, batch_size=self.batch_size, qat_epochs=self.qat_epochs, learning_rate=0.2,
                           cache_datasets=self.cache_datasets, approx=self.approx,
                           activation_quant_wait=self.activation_quant_wait, per_channel=self.per_channel,
                           symmetric=self.symmetric)

    def get_init_parents(self):
        return [{"quant_conf": [i for _ in range(self.quantizable_layers)]} for i in range(2, 9)]

    def crossover(self, parents):
        child_conf = [8 for _ in range(self.quantizable_layers)]
        for li in range(self.quantizable_layers):
            if random.random() < 0.95:  # 95 % probability of crossover
                child_conf[li] = random.choice(parents)["quant_conf"][li]
            else:
                child_conf[li] = 8

        if random.random() < 0.1:  # 10 % probability of mutation
            li = random.choice([x for x in range(self.quantizable_layers)])
            child_conf[li] = random.choice([2, 3, 4, 5, 6, 7, 8])

        return {"quant_conf": child_conf}


class QATAnalyzer(NSGAAnalyzer):
    def __init__(self, base_model, batch_size=64, qat_epochs=10, bn_freeze=25, learning_rate=0.05, warmup=0.0,
                 cache_datasets=False, approx=False, activation_quant_wait=0, per_channel=True, symmetric=True):
        self.base_model = base_model
        self.batch_size = batch_size
        self.qat_epochs = qat_epochs
        self.bn_freeze = bn_freeze
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.cache_datasets = cache_datasets
        self.approx = approx
        self.activation_quant_wait = activation_quant_wait
        self.per_channel = per_channel
        self.symmetric = symmetric
        self._mask = None

        self.ensure_cache_folder()

        # Current cache file
        i = 0
        while True:
            self.cache_file = "cache/%s_%d_%d_%d.json.gz" % ("mobilenet", batch_size, qat_epochs, i)
            if not os.path.isfile(self.cache_file):
                break
            i += 1

        print("Cache file: %s" % self.cache_file)
        self.cache = []
        self.load_cache()

    @staticmethod
    def ensure_cache_folder():
        if not os.path.exists("cache"):
            os.makedirs("cache")

    def load_cache(self):
        for fn in glob.glob("cache/%s_%d_%d_*.json.gz" % ("mobilenet", self.batch_size, self.qat_epochs)):
            if fn == self.cache_file:
                continue
            print("cache open", fn)

            act = json.load(gzip.open(fn))

            # find node in cache
            for c in act:
                conf = c["quant_conf"]

                # try to search in cache
                if not any(filter(lambda x: np.array_equal(x["quant_conf"], conf), self.cache)):
                    self.cache.append(c)

        tf.print("Cache loaded %d" % (len(self.cache)))

    def analyze(self, quant_configuration_set):
        for node_conf in quant_configuration_set:
            quant_conf = node_conf["quant_conf"]

            # try to search in cache
            cache_sel = self.cache.copy()

            # filter data
            for i in range(len(quant_conf)):
                cache_sel = filter(lambda x: x["quant_conf"][i] == quant_conf[i], cache_sel)
                cache_sel = list(cache_sel)

            # Get the accuracy
            if len(cache_sel) >= 1:  # Found in cache
                accuracy = cache_sel[0]["accuracy"]
                memory = cache_sel[0]["memory"]
                tf.print("Cache : %s;accuracy=%s;memory=%s;" % (str(quant_conf), accuracy, memory))
            else:  # Not found in cache
                quantized_model = self.quantize_model_by_config(quant_conf)

                accuracy = mobilenet_tinyimagenet_qat.main(q_aware_model=quantized_model,
                                                           epochs=self.qat_epochs,
                                                           bn_freeze=self.bn_freeze,
                                                           batch_size=self.batch_size,
                                                           learning_rate=self.learning_rate,
                                                           warmup=self.warmup,
                                                           checkpoints_dir=None,
                                                           logs_dir=None,
                                                           cache_dataset=True,
                                                           from_checkpoint=None,
                                                           verbose=False,
                                                           activation_quant_wait=self.activation_quant_wait
                                                           )

                # calculate size
                memory = calculate_model_size.calculate_weights_mobilenet_size(quantized_model)

            # Create output node
            node = node_conf.copy()
            node["quant_conf"] = quant_conf
            node["accuracy"] = float(accuracy)
            node["memory"] = int(memory)

            if len(cache_sel) == 0:  # If the data are not from the cache, cache it
                self.cache.append(node)
                json.dump(self.cache, gzip.open(self.cache_file, "wt", encoding="utf8"))

            yield node

    def __str__(self):
        return "cache(%s,%d)" % (self.cache_file, len(self.cache))

    def quantize_model_by_config(self, quant_config):
        quant_config = quant_config.copy()
        quant_config.append(8)  # Insert last value a default one, because 0 - 1 = -1 will be default

        final_quant_config = [quant_config[i - 1] for i in self.mask]
        config = [{"weight_bits": final_quant_config[i], "activation_bits": 8} for i in range(len(self.mask))]
        return quantize_model(self.base_model, config, approx=self.approx, per_channel=self.per_channel,
                              symmetric=self.symmetric)

    def get_quantizable_layers(self):
        return len(list(filter(lambda x: x != 0, self.mask)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self._get_quantizable_layers_mask()
        return self._mask

    def _get_quantizable_layers_mask(self):
        transformer = PerLayerQuantizeModelTransformer(self.base_model, [], {}, approx=self.approx,
                                                       per_channel=self.per_channel, symmetric=self.symmetric)

        groups = transformer.get_quantizable_layers_groups()
        mask = [0 for _ in range(len(groups))]
        count = 1
        for i, group in enumerate(groups):
            if calculate_model_size.calculate_weights_mobilenet_size(self.base_model, only_layers=group) > 0:
                mask[i] = count
                count = count + 1
        return mask
