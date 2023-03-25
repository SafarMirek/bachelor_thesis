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


class QATNSGA(NSGA):

    def __init__(self, logs_dir, base_model, parent_size=50, offspring_size=50, generations=25, batch_size=128,
                 qat_epochs=10):
        super().__init__(logs_dir=logs_dir,
                         parent_size=parent_size, offspring_size=offspring_size, generations=generations,
                         objectives=[("accuracy", True), ("memory", False)]
                         )
        self.base_model = base_model
        self.batch_size = batch_size
        self.qat_epochs = qat_epochs

        self.quantizable_layers = 37

    def get_maximal(self):
        return {
            "accuracy": 1.0,
            "memory": calculate_model_size.calculate_weights_mobilenet_size(self.base_model)
        }

    def init_analyzer(self) -> NSGAAnalyzer:
        return QATAnalyzer(self.base_model, batch_size=self.batch_size, qat_epochs=self.qat_epochs)

    def get_init_parents(self):
        return [{"quant_conf": [i for _ in range(self.quantizable_layers)]} for i in range(2, 9)]

    def crossover(self, parents):
        child_conf = [8 for _ in range(self.quantizable_layers)]
        for li in range(self.quantizable_layers):
            if random.random() < 0.90:  # 90 % probability of crossover
                child_conf[li] = random.choice(parents)["quant_conf"][li]
            else:
                child_conf[li] = 8

        if random.random() < 0.1:  # 10 % probability of mutation
            li = random.choice([x for x in range(self.quantizable_layers)])
            child_conf[li] = random.choice([2, 3, 4, 5, 6, 7, 8])

        return {"quant_conf": child_conf}


class QATAnalyzer(NSGAAnalyzer):
    def __init__(self, base_model, batch_size=64, qat_epochs=10):
        self.base_model = base_model
        self.batch_size = batch_size
        self.qat_epochs = qat_epochs

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
            # print(len(cache_sel))

            # filter data
            for i in range(len(quant_conf)):
                cache_sel = filter(lambda x: x["quant_conf"][i] == quant_conf[i], cache_sel)
                cache_sel = list(cache_sel)

            # Get the accuracy
            if len(cache_sel) >= 1:
                accuracy = cache_sel[0]["accuracy"]
                memory = cache_sel[0]["memory"]
                tf.print("Cache : %s;accuracy=%s;memory=%s;" % (str(quant_conf), accuracy, memory))
            else:
                quantized_model = self.quantize_model_by_config(quant_conf)

                accuracy = mobilenet_tinyimagenet_qat.main(q_aware_model=quantized_model,
                                                           epochs=self.qat_epochs,
                                                           bn_freeze=10e1000,
                                                           batch_size=self.batch_size,
                                                           learning_rate=0.01,
                                                           warmup=0.0,
                                                           checkpoints_dir=None,
                                                           logs_dir=None,
                                                           cache_dataset=False,
                                                           from_checkpoint=None,
                                                           verbose=False
                                                           )

                # calculate size
                memory = calculate_model_size.calculate_weights_mobilenet_size(quantized_model)

            # Create output node
            node = node_conf.copy()
            node["quant_conf"] = quant_conf
            node["accuracy"] = float(accuracy)
            node["memory"] = int(memory)

            if len(cache_sel) == 0:
                self.cache.append(node)
                json.dump(self.cache, gzip.open(self.cache_file, "wt", encoding="utf8"))

            yield node

    def __str__(self):
        return "cache(%s,%d)" % (self.cache_file, len(self.cache))

    def quantize_model_by_config(self, quant_config):
        config = [{"weight_bits": quant_config[i], "activation_bits": 8} for i in range(len(quant_config))]
        return quantize_model(self.base_model, config)
