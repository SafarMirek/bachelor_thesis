import abc
import gzip
import json
import os
import random

import tensorflow as tf
from paretoarchive.core import PyBspTreeArchive


class NSGAAnalyzer(abc.ABC):

    @abc.abstractmethod
    def analyze(self, configurations):
        """

        :param configurations: List of configurations that needs to be analyzes
        :return: List of configurations with added evaluation of each one
        """
        pass


class NSGA(abc.ABC):
    """
    Implementation of NSGA algorith
    """

    def __init__(self, logs_dir, parent_size=50, offspring_size=50, generations=25, objectives=None):
        if logs_dir is None:
            raise ValueError(f"Logs directory needs to be defined")

        if parent_size < 0:
            raise ValueError(f"Number of parents cannot be negative ({parent_size}<0)")

        if offspring_size < 0:
            raise ValueError(f"Number of offsprings cannot be negative ({offspring_size}<0)")

        if generations < 0:
            raise ValueError(f"Number of generations cannot be negative ({generations}<0)")

        if objectives is None:
            raise ValueError("Objectives need to be defined")

        self.logs_dir = logs_dir
        self.parent_size = parent_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.objectives = objectives

        self.analyzer = None

        self.ensure_logs_dir()

    def ensure_logs_dir(self):
        os.makedirs(self.logs_dir)

    def get_pareto_front(self, values):

        def map_obj_list(value):
            return [value[obj[0]] * (-1 if obj[1] else 1) for obj in self.objectives]

        pareto_ids = PyBspTreeArchive(2).filter([map_obj_list(x) for x in values],
                                                returnIds=True)

        return pareto_ids

    def run(self):
        """

        :return: final parents with their evaluation
        """
        start_gen = 0
        parents = self.get_init_parents()
        next_parents = list(self.get_analyzer().analyze(parents))

        offsprings = []

        for g in range(start_gen, self.generations + 1):
            print("Generation %d" % g)
            tf.print("generation:%d;cache=%s" % (g, str(self.get_analyzer())))
            # initial results from previous data:
            analyzed_offsprings = list(self.get_analyzer().analyze(offsprings))

            json.dump({"parent": parents, "offspring": offsprings},
                      gzip.open(self.logs_dir + "/run.%05d.json.gz" % g, "wt", encoding="utf8"))

            # reduce the number of elements
            filtered_results = next_parents + analyzed_offsprings
            next_parents = []
            missing = self.parent_size - len(next_parents)
            while missing > 0 and len(filtered_results) > 0:
                pareto_ids = self.get_pareto_front(filtered_results)
                pareto = [filtered_results[i] for i in pareto_ids]

                if len(pareto) <= missing:
                    next_parents += pareto
                else:  # distance crowding
                    next_parents += self.crowding_reduce(pareto, missing)

                for i in reversed(sorted(pareto_ids)):
                    filtered_results.pop(i)

                missing = self.parent_size - len(next_parents)

            # generate new candidate solutions:
            offsprings = self.generate_offsprings(parents=next_parents)

    def generate_offsprings(self, *, parents):
        """
        :param parents:
        :return: lists of generates offsprings
        """
        offsprings = []
        for i in range(0, self.offspring_size):
            # select two random parents
            parents = random.choices(parents, k=2)
            # generate offspring from these two parents
            offsprings.append(self.crossover(parents))

        return offsprings

    def crowding_distance(self, par):
        park = list(enumerate(par))
        distance = [0 for _ in range(len(par))]
        for obj in self.objectives:
            sorted_values = sorted(park, key=lambda x: x[1][obj])
            minval, maxval = 0, self.get_maximal()[obj]
            distance[sorted_values[0][0]] = float("inf")
            distance[sorted_values[-1][0]] = float("inf")

            for i in range(1, len(sorted_values) - 1):
                distance[sorted_values[i][0]] += abs(sorted_values[i - 1][1][obj] - sorted_values[i + 1][1][obj]) / (
                        maxval - minval)
        # print(distance)
        # print(sorted(distance, key=lambda x:-x))
        return zip(par, distance)

    def crowding_reduce(self, par, number):
        par = par
        while len(par) > number:
            vals = self.crowding_distance(par)
            vals = sorted(vals, key=lambda x: -x[1])  # sort by distance descending

            par = [x[0] for x in vals[:-1]]  # remove last
        return par

    def get_analyzer(self):
        if self.analyzer is None:
            self.analyzer = self.init_analyzer()
        return self.analyzer

    @abc.abstractmethod
    def crossover(self, parents):
        """Returns child"""
        pass

    @abc.abstractmethod
    def get_maximal(self):
        """Returns maximal values for objectives"""
        pass

    @abc.abstractmethod
    def init_analyzer(self) -> NSGAAnalyzer:
        """Returns analyzer"""
        pass

    @abc.abstractmethod
    def get_init_parents(self):
        pass
