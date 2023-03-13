# Author: Vojtěch Mrázek (mrazek@fit.vutbr.cz)

from __future__ import print_function
from paretoarchive import PyBspTreeArchive
import json, gzip
import random
import datetime, os
import argparse
import tensorflow as tf
import glob
from shutil import copyfile
import re


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
    def __init__(self, config, config_fn, data_dir, batch_size=1000, iterations=10):
        self.cache_prefix = os.path.basename(config_fn).replace(".pb", "")
        self.config = config
        self.data_dir = data_dir
        self.config_fn = config_fn
        self.batch_size = batch_size
        self.iterations = iterations

        i = 0
        while 1:
            self.cache_file = "cache/%s_%d_%d_%d.json.gz" % (self.cache_prefix, batch_size, iterations, i)
            if not os.path.isfile(self.cache_file): break
            i += 1

        print("Cache file: %s" % self.cache_file)
        self.cache = []
        self.refresh_cache()

    def refresh_cache(self):
        for fn in glob.glob("cache/%s_%d_%d_*.json.gz" % (self.cache_prefix, self.batch_size, self.iterations)):
            if fn == self.cache_file: continue
            print("cache open", fn)

            act = json.load(gzip.open(fn))

            # find node in cache
            for c in act:
                conf = c["multconf"]

                # try to search in cache
                cache_sel = self.cache.copy()
                # filter data and calculate the energy
                for l in self.config["layers"]:
                    cache_sel = filter(lambda x: x["multconf"][l] == conf[l], cache_sel)
                    cache_sel = list(cache_sel)

                if not cache_sel:
                    self.cache.append(c)

        tf.logging.info("Cache update %d" % (len(self.cache)))

    def analyze(self, config_set, update_iter=True, update_node=False):
        config = self.config

        if update_iter:
            self.refresh_cache()

        for allconf in config_set:
            conf = allconf["multconf"]

            # try to search in cache
            cache_sel = self.cache.copy()
            # print(len(cache_sel))
            power = 0.0
            sumlayers = 0

            # filter data and calculate the energy
            for l in config["layers"]:
                cache_sel = filter(lambda x: x["multconf"][l] == conf[l], cache_sel)
                cache_sel = list(cache_sel)

                power += config["power"][conf[l]] * config["layers"][l]
                sumlayers += config["layers"][l]

            power = float(power) / sumlayers

            # Get the accuracy
            if len(cache_sel) >= 1:
                accuracy = cache_sel[0]["accuracy"]
                tf.logging.info("Cache : %s;accuracy=%s;" % (str(conf), accuracy))
            else:
                # assert False # todo run inference and save to cache
                accuracy = cifar10_ax_inference.main(data_dir=self.data_dir, config=self.config_fn,
                                                     iterations=self.iterations, batch_size=self.batch_size, mult=conf,
                                                     tune=True)

            # Create output node
            node = allconf.copy()
            node["multconf"] = conf
            node["accuracy"] = accuracy
            node["power"] = power

            if not len(cache_sel):
                self.cache.append(node)
                # print(json.dumps(cache)) #, gzip.open("cache.json.gz", "w"))
                json.dump(self.cache, gzip.open(self.cache_file, "wt", encoding="utf8"))

            if update_node:
                self.refresh_cache()

            yield node

    def __str__(self):
        return "cache(%s,%d)" % (self.cache_file, len(self.cache))
