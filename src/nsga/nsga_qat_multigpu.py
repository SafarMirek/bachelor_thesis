# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import glob
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor

import keras
import numpy as np
import tensorflow as tf

import calculate_model_size
import mobilenet_tinyimagenet_qat
import nsga.nsga_qat
from nsga.nsga import NSGAAnalyzer
from tf_quantization.quantize_model import quantize_model
from tf_quantization.transforms.quantize_transforms import PerLayerQuantizeModelTransformer

from queue import Queue


class MultiGPUQATNSGA(nsga.nsga_qat.QATNSGA):

    def __init__(self, logs_dir, base_model, parent_size=50, offspring_size=50, generations=25, batch_size=128,
                 qat_epochs=10, previous_run=None, cache_datasets=False, approx=False, activation_quant_wait=0,
                 per_channel=True, symmetric=True, learning_rate=0.2):
        super().__init__(logs_dir, base_model, parent_size, offspring_size, generations, batch_size, qat_epochs,
                         previous_run, cache_datasets, approx, activation_quant_wait, per_channel, symmetric,
                         learning_rate)

    def init_analyzer(self) -> NSGAAnalyzer:
        logs_dir_pattern = os.path.join(self.logs_dir, "logs/%s")
        checkpoints_dir_pattern = os.path.join(self.logs_dir, "checkpoints/%s")
        return MultiGPUQATAnalyzer(batch_size=self.batch_size, qat_epochs=self.qat_epochs,
                                   learning_rate=self.learning_rate, cache_datasets=self.cache_datasets,
                                   approx=self.approx, activation_quant_wait=self.activation_quant_wait,
                                   per_channel=self.per_channel, symmetric=self.symmetric,
                                   logs_dir_pattern=logs_dir_pattern,
                                   checkpoints_dir_pattern=checkpoints_dir_pattern)


class MultiGPUQATAnalyzer(NSGAAnalyzer):
    def __init__(self, batch_size=64, qat_epochs=10, bn_freeze=25, learning_rate=0.05, warmup=0.0,
                 cache_datasets=False, approx=False, activation_quant_wait=0, per_channel=True, symmetric=True,
                 logs_dir_pattern=None, checkpoints_dir_pattern=None):
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
        self.logs_dir_pattern = logs_dir_pattern
        self.checkpoints_dir_pattern = checkpoints_dir_pattern
        self._mask = None
        self._queue = None

        self.ensure_cache_folder()

        # Current cache file
        i = 0
        while True:
            self.cache_file = "cache/%s_%d_%d_%d_%.5f_%.2f_%d_%r_%r_%r_%d.json.gz" % (
                "mobilenet", batch_size, qat_epochs, bn_freeze, learning_rate, warmup, activation_quant_wait, approx,
                per_channel, symmetric,
                i)
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
        for fn in glob.glob("cache/%s_%d_%d_%d_%.5f_%.2f_%d_%r_%r_%r_*.json.gz" % (
                "mobilenet", self.batch_size, self.qat_epochs, self.bn_freeze, self.learning_rate, self.warmup,
                self.activation_quant_wait, self.approx,
                self.per_channel, self.symmetric)):
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

    @property
    def queue(self):
        if self._queue is None:
            self._queue = Queue()
            logical_devices = tf.config.list_logical_devices('GPU')
            for device in logical_devices:
                self._queue.put(device)
        return self._queue

    def analyze(self, quant_configuration_set):
        needs_eval = []

        for node_conf in quant_configuration_set:
            quant_conf = node_conf["quant_conf"]

            if not any(x["quant_conf"] == quant_conf for x in self.cache):  # not in cache
                if not any(conf == quant_conf for conf in needs_eval):  # not in needs eval
                    needs_eval.append(quant_conf)

        # Run evaluation for all configurations that needs it
        logical_devices = tf.config.list_logical_devices('GPU')
        pool = ThreadPoolExecutor(max_workers=len(logical_devices))

        print("Needs eval: " + str(needs_eval) + " on " + str(len(logical_devices)) + " GPUs.")
        results = list(pool.map(self.get_eval_of_config, needs_eval))
        print("Eval done")

        for i, quant_conf in enumerate(needs_eval):
            node = {
                "quant_conf": quant_conf,
                "accuracy": float(results[i][0]),
                "memory": int(results[i][1])
            }
            self.cache.append(node)
            json.dump(self.cache, gzip.open(self.cache_file, "wt", encoding="utf8"))

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
                raise ValueError("All configurations should be in cache, how has this happends?")

            # Create output node
            node = node_conf.copy()
            node["quant_conf"] = quant_conf
            node["accuracy"] = float(accuracy)
            node["memory"] = int(memory)

            yield node

    def __str__(self):
        return "cache(%s,%d)" % (self.cache_file, len(self.cache))

    def get_number_of_quantizable_layers(self):
        return len(list(filter(lambda x: x != 0, self.mask)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self._get_quantizable_layers_mask()
        return self._mask

    def _get_quantizable_layers_mask(self):
        base_model = keras.models.load_model("mobilenet_tinyimagenet.keras")
        transformer = PerLayerQuantizeModelTransformer(base_model, [], {}, approx=self.approx,
                                                       per_channel=self.per_channel,
                                                       symmetric=self.symmetric)

        groups = transformer.get_quantizable_layers_groups()
        mask = [-1 for _ in range(len(groups))]
        count = 0
        for i, group in enumerate(groups):
            if calculate_model_size.calculate_weights_mobilenet_size(base_model, only_layers=group,
                                                                     per_channel=self.per_channel,
                                                                     symmetric=self.symmetric) > 0:
                mask[i] = count
                count = count + 1
        return mask

    def apply_mask(self, chromosome):
        quant_config = chromosome.copy()
        quant_config.append(8)
        final_quant_config = [quant_config[i] for i in self.mask]
        config = [
            {
                "weight_bits": final_quant_config[i],
                "activation_bits": 8
            } for i in range(len(self.mask))
        ]
        return config

    def quantize_model_by_config(self, quant_config):
        config = self.apply_mask(quant_config)
        base_model = keras.models.load_model("mobilenet_tinyimagenet.keras")
        return quantize_model(base_model, config, approx=self.approx, per_channel=self.per_channel,
                              symmetric=self.symmetric)

    def get_eval_of_config(self, quant_config):
        device = self.queue.get()
        try:
            print(f"Quant config {quant_config} is going to be evaluated on {device.name}")
            with tf.device(device.name):
                quantized_model = self.quantize_model_by_config(quant_config)

                checkpoints_dir = None
                if self.checkpoints_dir_pattern is not None:
                    checkpoints_dir = self.checkpoints_dir_pattern % '_'.join(map(lambda x: str(x), quant_config))

                logs_dir = None
                if self.logs_dir_pattern is not None:
                    logs_dir = self.logs_dir_pattern % '_'.join(map(lambda x: str(x), quant_config))

                accuracy = mobilenet_tinyimagenet_qat.main(q_aware_model=quantized_model,
                                                           epochs=self.qat_epochs,
                                                           eval_epochs=50,
                                                           bn_freeze=self.bn_freeze,
                                                           batch_size=self.batch_size,
                                                           learning_rate=self.learning_rate,
                                                           warmup=self.warmup,
                                                           checkpoints_dir=checkpoints_dir,
                                                           logs_dir=logs_dir,
                                                           cache_dataset=self.cache_datasets,
                                                           from_checkpoint=None,
                                                           verbose=False,
                                                           activation_quant_wait=self.activation_quant_wait,
                                                           save_best_only=True
                                                           )
                # calculate size
                memory = calculate_model_size.calculate_weights_mobilenet_size(quantized_model,
                                                                               per_channel=self.per_channel,
                                                                               symmetric=self.symmetric)

                return accuracy, memory
        finally:
            self.queue.put(device)
