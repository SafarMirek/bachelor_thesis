import glob
import gzip
import json
import os

import keras
import numpy as np
import tensorflow as tf

import calculate_model_size
import mobilenet_tinyimagenet_qat
import nsga.nsga_qat
from nsga.nsga import NSGAAnalyzer
from tf_quantization.quantize_model import quantize_model
from tf_quantization.transforms.quantize_transforms import PerLayerQuantizeModelTransformer

from multiprocessing import Pool, current_process, Queue


class QATNSGA(nsga.nsga_qat.QATNSGA):

    def __init__(self, logs_dir, base_model, parent_size=50, offspring_size=50, generations=25, batch_size=128,
                 qat_epochs=10, previous_run=None):
        super().__init__(logs_dir, base_model, parent_size, offspring_size, generations, batch_size, qat_epochs,
                         previous_run)

    def init_analyzer(self) -> NSGAAnalyzer:
        return MultiGPUQATAnalyzer(batch_size=self.batch_size, qat_epochs=self.qat_epochs,
                                   learning_rate=0.2)


class MultiGPUQATAnalyzer(NSGAAnalyzer):
    def __init__(self, batch_size=64, qat_epochs=10, bn_freeze=25, learning_rate=0.05, warmup=0.0):
        self.batch_size = batch_size
        self.qat_epochs = qat_epochs
        self.bn_freeze = bn_freeze
        self.learning_rate = learning_rate
        self.warmup = warmup
        self._mask = None
        self._queue = None

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
        pool = Pool(processes=len(logical_devices))

        results = pool.map(self.get_eval_of_config, needs_eval)
        pool.close()

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

    def get_quantizable_layers(self):
        return len(list(filter(lambda x: x != 0, self.mask)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self._get_quantizable_layers_mask()
        return self._mask

    def _get_quantizable_layers_mask(self):
        base_model = keras.models.load_model("mobilenet_tinyimagenet.keras")
        transformer = PerLayerQuantizeModelTransformer(base_model, [], {})

        groups = transformer.get_quantizable_layers_groups()
        mask = [0 for _ in range(len(groups))]
        count = 1
        for i, group in enumerate(groups):
            if calculate_model_size.calculate_weights_mobilenet_size(base_model, only_layers=group) > 0:
                mask[i] = count
                count = count + 1
        return mask

    def quantize_model_by_config(self, quant_config):
        quant_config = quant_config.copy()
        quant_config.append(8)  # Insert last value a default one, because 0 - 1 = -1 will be default

        final_quant_config = [quant_config[i - 1] for i in self.mask]
        config = [{"weight_bits": final_quant_config[i], "activation_bits": 8} for i in range(len(self.mask))]
        base_model = keras.models.load_model("mobilenet_tinyimagenet.keras")
        return quantize_model(base_model, config)

    def get_eval_of_config(self, quant_config):
        device = self.queue.get()
        try:
            print(f"Quant config {quant_config} is going to be evaluated on {device.name}")
            with tf.device(device.name):
                quantized_model = self.quantize_model_by_config(quant_config)

                accuracy = mobilenet_tinyimagenet_qat.main(q_aware_model=quantized_model,
                                                           epochs=self.qat_epochs,
                                                           bn_freeze=self.bn_freeze,
                                                           batch_size=self.batch_size,
                                                           learning_rate=self.learning_rate,
                                                           warmup=self.warmup,
                                                           checkpoints_dir=None,
                                                           logs_dir=None,
                                                           cache_dataset=False,
                                                           from_checkpoint=None,
                                                           verbose=False
                                                           )
                # calculate size
                memory = calculate_model_size.calculate_weights_mobilenet_size(quantized_model)

                return accuracy, memory
        finally:
            self.queue.put(device)
