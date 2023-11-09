# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import abc
import datetime
import glob
import gzip
import json
import os
import random
import re
from shutil import copyfile

import tensorflow as tf
from paretoarchive.core import PyBspTreeArchive


class NSGAAnalyzer(abc.ABC):
    """
    Analyzer for NSGA-II to evaluate chromosomes before parents selection
    """

    @abc.abstractmethod
    def analyze(self, configurations):
        """
        This method analyzes chromosomes and returns them with their evaluation
        :param configurations: List of configurations that needs to be analyzes
        :return: List of configurations with added evaluation of each one
        """
        pass


class NSGAState:
    """
    State of the NSGA-II
    """

    def __init__(self, generation=None, parents=None, offsprings=None):
        """
        Constructs NSGAState

        :param generation: Current generation
        :param parents: Current parents
        :param offsprings: Current offsprings list
        """
        self._generation = generation
        self._parents = parents
        self._offsprings = offsprings
        self._restored = False

    def get_generation(self):
        """
        Returns current number of generationm
        :return: Number of generation
        """
        return self._generation

    def get_parents(self):
        """
        Return current parents
        :return: List of parents
        """
        return self._parents

    def get_offsprings(self):
        """
        Get current offsprings
        :return: List of offsprings
        """
        return self._offsprings

    def save_to(self, logs_dir):
        """
        Saves state to file
        :param logs_dir: Path to logs directory
        """
        if self._restored:
            return
        json.dump({"parent": self._parents, "offspring": self._offsprings},
                  gzip.open(logs_dir + "/run.%05d.json.gz" % self._generation, "wt", encoding="utf8"))

    def set_offsprings(self, new_offsprings):
        """
        Sets offsprings in the state
        :param new_offsprings: List of offsprings that replaces current list of offsprings
        """
        self._offsprings = new_offsprings

    @classmethod
    def restore_from(cls, run_file: str):
        print("# loading %s" % run_file)
        pr = json.load(gzip.open(run_file))
        parents = pr["parent"]
        offsprings = pr["offspring"]
        generation = int(re.match(r".*run\.(\d+).json.gz", run_file).group(1))
        tf.print(f"Restored generation {generation} with {len(parents)} parents and {len(offsprings)} offsprings")
        res_state = cls(generation=generation, parents=parents, offsprings=offsprings)
        res_state._restored = True
        return res_state


class NSGA(abc.ABC):
    """
    Implementation of NSGA algorith
    """

    def __init__(self, logs_dir, parent_size=50, offspring_size=50, generations=25, objectives=None, previous_run=None):
        """
        Constructs NSGA

        :param logs_dir: Path to log directory
        :param parent_size: Number of parents
        :param offspring_size: number of offsprings
        :param generations: Number of generations
        :param objectives: List of watched objectives
        :param previous_run: Path to previous run to restore from there
        """
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

        self.state = None
        if previous_run is not None:
            self._restore_state(previous_run)

        self.ensure_logs_dir()

        if previous_run is None:
            self._check_if_empty()

    def _restore_state(self, previous_run):
        """
        Restores state from previous run
        :param previous_run: Path to previous run log dir
        """
        df = glob.glob(previous_run + "/run.*.gz")
        if self.logs_dir != previous_run:
            for d in df:
                copyfile(d, self.logs_dir + "/" + os.path.basename(d))
                print("# file %s copied" % d)
        df = sorted(df)
        d = df[-1]
        self.state = NSGAState.restore_from(run_file=d)

    def ensure_logs_dir(self):
        """
        Ensures logs directory is created
        """
        try:
            os.makedirs(self.logs_dir)
        except FileExistsError:
            pass  # Folder already exists no need to create it

    def _check_if_empty(self):
        files = os.listdir(self.logs_dir)
        if len(files) > 0:
            print("ERROR: Folder for new run is not empty")
            exit(1)

    def _generate_run_information(self):
        print("Generation configuration information to " + os.path.abspath(self.logs_dir + "/configuration.json"))
        run_info = {
            "start_time": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "configuration": self.get_configuration()
        }
        with open(self.logs_dir + "/configuration.json", "w") as outfile:
            json.dump(run_info, outfile)

    @abc.abstractmethod
    def get_configuration(self):
        pass

    def get_pareto_front(self, values):
        """
        Returns pareto front from values
        :param values: Data
        :return: pareto front from data by watched objectives (list of indexes in original data)
        """

        def map_obj_list(value):
            return [value[obj[0]] * (-1 if obj[1] else 1) for obj in self.objectives]

        pareto_ids = PyBspTreeArchive(len(self.objectives)).filter([map_obj_list(x) for x in values],
                                                returnIds=True)

        return pareto_ids

    def get_current_state(self) -> NSGAState:
        """
        Get current state of NSGA
        :return: Current state of NSGA
        """
        return self.state

    def run_next_generation(self):
        """
        Run one NSGA generation
        """
        current_state = self.get_current_state()
        g = self.get_current_state().get_generation()
        print("Generation %d" % g)
        tf.print("generation:%d;cache=%s" % (g, str(self.get_analyzer())))
        # initial results from previous data:
        analyzed_offsprings = list(self.get_analyzer().analyze(current_state.get_offsprings()))
        current_state.set_offsprings(analyzed_offsprings)
        current_state.save_to(logs_dir=self.logs_dir)

        # reduce the number of elements
        filtered_results = current_state.get_parents() + current_state.get_offsprings()
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

        # generate new candidate solutions
        offsprings = self.generate_offsprings(parents=next_parents)

        # set new state
        self.state = NSGAState(generation=g + 1, parents=next_parents, offsprings=offsprings)

    def run(self):
        """
        Runs specified number of generations
        """
        if self.state is None:
            self._generate_run_information()

            parents = self.get_init_parents()
            next_parents = list(self.get_analyzer().analyze(parents))
            self.state = NSGAState(generation=0, parents=next_parents, offsprings=[])

        while self.get_current_state().get_generation() <= self.generations:
            self.run_next_generation()

    def generate_offsprings(self, *, parents):
        """
        Generate offsprings from parents using crossover and mutation
        :param parents: List of parents
        :return: list of generated offsprings
        """
        offsprings = []
        for i in range(0, self.offspring_size):
            # select two random parents
            selected_parents = random.sample(parents, k=2)
            # generate offspring from these two parents
            offsprings.append(self.crossover(selected_parents))

        return offsprings

    def crowding_distance(self, pareto_front):
        """
        Calculates crowding distance for each individual
        :param pareto_front: Set of individuals
        :return: list of pairs of individual and its crowding distance
        """
        park = list(enumerate(pareto_front))
        distance = [0 for _ in range(len(pareto_front))]
        for obj, asc in self.objectives:
            sorted_values = sorted(park, key=lambda x: x[1][obj])
            min_val, max_val = 0, self.get_maximal()[obj]
            distance[sorted_values[0][0]] = float("inf")
            distance[sorted_values[-1][0]] = float("inf")

            for i in range(1, len(sorted_values) - 1):
                distance[sorted_values[i][0]] += abs(sorted_values[i - 1][1][obj] - sorted_values[i + 1][1][obj]) / (
                        max_val - min_val)
        return zip(pareto_front, distance)

    def crowding_reduce(self, pareto_front, number):
        """
        Reduces pareto front to only <number> of individuals based on crowding distance
        :param pareto_front: Set of individuals
        :param number: Required number of individuals
        :return: reduced set
        """
        pareto_front = pareto_front
        while len(pareto_front) > number:
            vals = self.crowding_distance(pareto_front)
            vals = sorted(vals, key=lambda x: -x[1])  # sort by distance descending

            pareto_front = [x[0] for x in vals[:-1]]  # remove last
        return pareto_front

    def get_analyzer(self):
        """
        Get analyzer, if it was not created, initialize it
        :return: NSGAAnalyzer
        """
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
        """
        Returns initial parents for first population
        """
        pass
