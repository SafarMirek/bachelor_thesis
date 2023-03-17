# Author: Vojtěch Mrázek (mrazek@fit.vutbr.cz)
# Edited by Miroslav Šafář (xsafar23@fit.vutbr.cz)

import numpy as np
import json, gzip
import os
import tensorflow as tf
import glob

import calculate_model_size
import cifar10_qat_evaluate
from tf_quantization.quantize_model import quantize_model


def crowding_distance(par, maximal, objs=["accuracy", "memory"]):
    park = list(zip(range(len(par)), par))
    distance = [0 for i in range(len(par))]
    for o in objs:
        sval = sorted(park, key=lambda x: x[1][o])
        minval, maxval = 0, maximal[o]
        distance[sval[0][0]] = float("inf")
        distance[sval[-1][0]] = float("inf")

        for i in range(1, len(sval) - 1):
            distance[sval[i][0]] += abs(sval[i - 1][1][o] - sval[i + 1][1][o]) / (maxval - minval)
    # print(distance)
    # print(sorted(distance, key=lambda x:-x))
    return zip(par, distance)


def crowding_reduce(par, number, maximal):
    par = par
    while len(par) > number:
        vals = crowding_distance(par, maximal)
        vals = sorted(vals, key=lambda x: -x[1])  # sort by distance descending
        # print(vals)

        par = [x[0] for x in vals[:-1]]
    return par


class Analyzer:
    def __init__(self, base_model, batch_size=64, qat_epochs=10, pretrained_qat_weights_path=None):
        self.base_model = base_model
        self.batch_size = batch_size
        self.qat_epochs = qat_epochs
        self.pretrained_qat_weights_path = pretrained_qat_weights_path

        # Current cache file
        i = 0
        while 1:
            self.cache_file = "cache/%s_%d_%d_%d.json.gz" % ("resnet8", batch_size, qat_epochs, i)
            if not os.path.isfile(self.cache_file): break
            i += 1

        print("Cache file: %s" % self.cache_file)
        self.cache = []
        self.load_cache()

    def load_cache(self):
        for fn in glob.glob("cache/%s_%d_%d_*.json.gz" % ("resnet8", self.batch_size, self.qat_epochs)):
            if fn == self.cache_file: continue
            print("cache open", fn)

            act = json.load(gzip.open(fn))

            # find node in cache
            for c in act:
                conf = c["quant_conf"]

                # try to search in cache
                if not any(filter(lambda x: np.array_equal(x["quant_conf"], conf), self.cache)):
                    self.cache.append(c)

        tf.logging.info("Cache loaded %d" % (len(self.cache)))

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
                tf.logging.info("Cache : %s;accuracy=%s;memory=%s;" % (str(quant_conf), accuracy, memory))
            else:
                quantized_model = self.quantize_model_by_config(quant_conf)

                # TODO: Run QAT for some amount of epochs and then evaluate the accurancy
                accuracy = cifar10_qat_evaluate.main(q_aware_model=quantized_model,
                                                     epochs=self.qat_epochs,
                                                     bn_freeze=10e1000,
                                                     batch_size=64,
                                                     learning_rate=0.001,
                                                     warmup=0.0,
                                                     from_checkpoint=self.pretrained_qat_weights_path
                                                     )

                # calculate size
                memory = calculate_model_size.calculate_weights_mobilenet_size(quantized_model)

            # Create output node
            node = node_conf.copy()
            node["quant_conf"] = quant_conf
            node["accuracy"] = accuracy
            node["memory"] = memory

            if len(cache_sel) == 0:
                self.cache.append(node)
                json.dump(self.cache, gzip.open(self.cache_file, "wt", encoding="utf8"))

            yield node

    def __str__(self):
        return "cache(%s,%d)" % (self.cache_file, len(self.cache))

    def quantize_model_by_config(self, quant_config):
        config = [{"weight_bits": quant_config[i], "activation_bits": 8} for i in range(len(quant_config))]
        return quantize_model(self.base_model, config)
