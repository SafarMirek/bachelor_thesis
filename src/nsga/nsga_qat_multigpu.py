# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import tensorflow as tf

import mobilenet_tinyimagenet_qat
import nsga.nsga_qat
import resnet_cifar_qat
from nsga.nsga import NSGAAnalyzer

from queue import Queue

from mapper_facade import MapperFacade


class MultiGPUQATNSGA(nsga.nsga_qat.QATNSGA):
    """
    NSGA-II for proposed system, it uses MultiGPU QAT Analyzer for evaluation of individuals
    """

    def __init__(self, logs_dir, base_model_path, parent_size=50, offspring_size=50, generations=25, batch_size=128,
                 qat_epochs=10, previous_run=None, cache_datasets=False, approx=False, activation_quant_wait=0,
                 per_channel=True, symmetric=True, learning_rate=0.2, timeloop_heuristic="random",
                 timeloop_architecture="eyeriss", model_name="mobilenet", bn_freeze=25):
        super().__init__(logs_dir, base_model_path, parent_size, offspring_size, generations, batch_size, qat_epochs,
                         previous_run, cache_datasets, approx, activation_quant_wait, per_channel, symmetric,
                         learning_rate, timeloop_heuristic, timeloop_architecture, model_name, bn_freeze)

    def init_analyzer(self) -> NSGAAnalyzer:
        # logs_dir_pattern = os.path.join(self.logs_dir, "logs/%s")
        logs_dir_pattern = None
        checkpoints_dir_pattern = os.path.join(self.logs_dir, "checkpoints/%s")
        return MultiGPUQATAnalyzer(batch_size=self.batch_size, qat_epochs=self.qat_epochs,
                                   learning_rate=self.learning_rate, cache_datasets=self.cache_datasets,
                                   approx=self.approx, activation_quant_wait=self.activation_quant_wait,
                                   per_channel=self.per_channel, symmetric=self.symmetric,
                                   logs_dir_pattern=logs_dir_pattern,
                                   checkpoints_dir_pattern=checkpoints_dir_pattern,
                                   base_model_path=self.base_model_path, timeloop_heuristic=self.timeloop_heuristic,
                                   timeloop_architecture=self.timeloop_architecture, model_name=self.model_name,
                                   bn_freeze=self.bn_freeze)


class MultiGPUQATAnalyzer(nsga.nsga_qat.QATAnalyzer):
    """
    Analyzer for QATNSGA

    This analyzer analyzes individuals by running a few epochs using quantization-aware training
    and tracking best achieved Top-1 accuracy

    This analyzer supports Multi GPU evaluation,
    multiple configurations is evaluated at the same time on multiple GPUs
    """

    def __init__(self, base_model_path, batch_size=64, qat_epochs=10, bn_freeze=25, learning_rate=0.05, warmup=0.0,
                 cache_datasets=False, approx=False, activation_quant_wait=0, per_channel=True, symmetric=True,
                 logs_dir_pattern=None, checkpoints_dir_pattern=None, timeloop_heuristic="exhaustive",
                 timeloop_architecture="eyeriss", include_timeloop_dump=False, model_name="mobilenet"):
        super().__init__(base_model_path, batch_size, qat_epochs, bn_freeze, learning_rate, warmup, cache_datasets,
                         approx,
                         activation_quant_wait, per_channel, symmetric, logs_dir_pattern, checkpoints_dir_pattern,
                         timeloop_heuristic, timeloop_architecture, include_timeloop_dump=include_timeloop_dump,
                         model_name=model_name)
        self._queue = None
        self._timeloop_pool = ThreadPoolExecutor(max_workers=1)

        self._lock = Lock()

    @property
    def queue(self):
        if self._queue is None:
            self._queue = Queue()
            logical_devices = tf.config.list_logical_devices('GPU')
            for device in logical_devices:
                self._queue.put(device)
        return self._queue

    def update_cache(self, node_to_update):
        self._lock.acquire()

        self.cache.clear()
        self.load_cache()

        quant_conf = node_to_update["quant_conf"]

        if not any(x["quant_conf"] == quant_conf for x in self.cache):
            self.cache.append(node_to_update)
        else:
            cached_entry = list(filter(lambda x: x["quant_conf"] == quant_conf, self.cache))[0]
            for key in node_to_update:
                if key not in cached_entry:
                    print(f"Updating {key} in {cached_entry['quant_conf']} to {node_to_update[key]}")
                    cached_entry[key] = node_to_update[key]
                else:
                    print(f"{key} is already in {cached_entry['quant_conf']}")

        json.dump(self.cache, gzip.open(self.cache_file, "wt", encoding="utf8"))

        self._lock.release()

    def read_from_cache(self, quant_config):
        self._lock.acquire()

        if not any(x["quant_conf"] == quant_config for x in self.cache):
            node = None
        else:
            node = list(filter(lambda x: x["quant_conf"] == quant_config, self.cache))[0]

        self._lock.release()

        return node

    def analyze(self, quant_configuration_set):
        """
        Analyze configurations on multiple GPUs

        This analyzer uses all available GPU to make evaluation of configurations

        :param quant_configuration_set: List of configurations for evaluation
        :return: Analyzer list of configurations
        """
        needs_eval = []

        for node_conf in quant_configuration_set:
            quant_conf = node_conf["quant_conf"]

            entry = self.read_from_cache(quant_conf)

            if entry is None:
                entry = {"quant_conf": quant_conf}

            needs_eval.append(entry)

        # Run evaluation for all configurations that needs it
        logical_devices = tf.config.list_logical_devices('GPU')
        pool = ThreadPoolExecutor(max_workers=len(logical_devices))  # One thread for timeloop

        print("Needs eval: " + str(needs_eval) + " on " + str(len(logical_devices)) + " GPUs.")
        results = list(pool.map(self.get_eval_of_config, needs_eval))
        print("Eval done")

        for i, quant_conf in enumerate(needs_eval):
            node = {
                "quant_conf": quant_conf["quant_conf"],
                "accuracy": float(results[i][0]),
                f"total_edp_{self.timeloop_architecture}": float(results[i][1].result()[f"total_edp"]),
                f"total_cycles_{self.timeloop_architecture}": int(results[i][1].result()[f"total_cycles"]),
                f"total_energy_{self.timeloop_architecture}": float(results[i][1].result()[f"total_energy"]),
                f"total_lastlevelaccesses_{self.timeloop_architecture}": int(
                    results[i][1].result()[f"total_lastlevelaccesses"]),
            }
            if self.include_timeloop_dump:
                node[f"timeloop_dump_{self.timeloop_architecture}"] = results[i][1].result()["timeloop_dump"]

            self.update_cache(node)

        for node_conf in quant_configuration_set:
            quant_conf = node_conf["quant_conf"]

            cached_entry = self.read_from_cache(quant_conf)

            # Get the accuracy
            if cached_entry is not None:  # Found in cache
                accuracy = cached_entry["accuracy"]
                total_edp = cached_entry[f"total_edp_{self.timeloop_architecture}"]
                total_cycles = cached_entry[f"total_cycles_{self.timeloop_architecture}"]
                total_energy = cached_entry[f"total_energy_{self.timeloop_architecture}"]
                total_lastlevelaccesses = cached_entry[f"total_lastlevelaccesses_{self.timeloop_architecture}"]
                if self.include_timeloop_dump:
                    timeloop_dump = cached_entry[f"timeloop_dump_{self.timeloop_architecture}"]
                else:
                    timeloop_dump = None
                # total_cycles = cache_sel[0]["total_cycles"]
                tf.print(f"Cache : %s;accuracy=%s;edp_{self.timeloop_architecture}=%s;" % (
                    str(quant_conf), accuracy, total_edp))
            else:  # Not found in cache
                raise ValueError(f"All configurations should be in cache, how has this happends ({quant_conf})")

            # Create output node
            node = node_conf.copy()
            node["quant_conf"] = quant_conf
            node["accuracy"] = float(accuracy)
            node[f"total_edp"] = float(total_edp)
            node[f"total_cycles"] = int(total_cycles)
            node[f"total_energy"] = float(total_energy)
            node[f"total_lastlevelaccesses"] = int(total_lastlevelaccesses)
            if self.include_timeloop_dump:
                node[f"timeloop_dump"] = timeloop_dump
            # node["total_cycles"] = int(total_cycles)

            yield node

    def eval_param(self, eval_param, quantized_model, quant_config, device_name):
        if eval_param == "hardware_params":
            collected_params = [f"total_edp_{self.timeloop_architecture}",
                                f"total_energy_{self.timeloop_architecture}",
                                f"total_cycles_{self.timeloop_architecture}",
                                f"total_lastlevelaccesses_{self.timeloop_architecture}"
                                ]
            if all(x in quant_config for x in collected_params) and (
                    not self.include_timeloop_dump or f"timeloop_dump_{self.timeloop_architecture}" in collected_params
            ):
                result = {
                    f"total_edp": quant_config[f"total_edp_{self.timeloop_architecture}"],
                    f"total_energy": quant_config[f"total_energy_{self.timeloop_architecture}"],
                    f"total_cycles": quant_config[f"total_cycles_{self.timeloop_architecture}"],
                    f"total_lastlevelaccesses": quant_config[f"total_lastlevelaccesses_{self.timeloop_architecture}"]
                }
                if self.include_timeloop_dump:
                    result["timeloop_dump"] = quant_config[f"timeloop_dump_{self.timeloop_architecture}"]
                return result

            mapper_facade = MapperFacade(architecture=self.timeloop_architecture)
            total_valid = 0 if self.timeloop_heuristic == "exhaustive" else 30000

            input_size = "32,32,3" if self.model_name == "resnet" else "224,224,3"

            hardware_params = mapper_facade.get_hw_params_parse_model(model=self.base_model_path, batch_size=1,
                                                                      bitwidths=nsga.nsga_qat.get_config_from_model(
                                                                          quantized_model),
                                                                      input_size=input_size, threads=24,
                                                                      heuristic=self.timeloop_heuristic,
                                                                      metrics=("edp", ""),
                                                                      total_valid=total_valid,
                                                                      verbose=True)

            total_edp = sum(map(lambda x: float(x["EDP [J*cycle]"]), hardware_params.values()))
            total_cycles = sum(map(lambda x: int(x["Cycles"]), hardware_params.values()))
            total_energy = sum(map(lambda x: float(x["Energy [uJ]"]), hardware_params.values()))
            total_lastlevelaccesses = sum(map(lambda x: int(x["LastLevelAccesses"]), hardware_params.values()))

            result = {
                f"total_edp": total_edp,
                f"total_energy": total_energy,
                f"total_cycles": total_cycles,
                f"total_lastlevelaccesses": total_lastlevelaccesses
            }

            if self.include_timeloop_dump:
                result["timeloop_dump"] = hardware_params

            return result

        elif eval_param == "accuracy":
            if "accuracy" in quant_config:
                return quant_config["accuracy"]

            with tf.device(device_name):
                checkpoints_dir = None
                if self.checkpoints_dir_pattern is not None:
                    checkpoints_dir = self.checkpoints_dir_pattern % '_'.join(
                        map(lambda x: str(x), quant_config["quant_conf"]))

                logs_dir = None
                if self.logs_dir_pattern is not None:
                    logs_dir = self.logs_dir_pattern % '_'.join(map(lambda x: str(x), quant_config["quant_conf"]))

                if self.model_name == "mobilenet":
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
                elif self.model_name == "resnet":
                    accuracy = resnet_cifar_qat.main(q_aware_model=quantized_model,
                                                     epochs=self.qat_epochs,
                                                     bn_freeze=self.bn_freeze,
                                                     batch_size=self.batch_size,
                                                     learning_rate=self.learning_rate,
                                                     checkpoints_dir=checkpoints_dir,
                                                     logs_dir=logs_dir,
                                                     cache_dataset=self.cache_datasets,
                                                     from_checkpoint=None,
                                                     verbose=False,
                                                     activation_quant_wait=self.activation_quant_wait,
                                                     save_best_only=True
                                                     )
                else:
                    accuracy = 0
                return accuracy

    def get_eval_of_config(self, quant_config):
        """
        Get evaluation for configuration
        :param quant_config:  to be evaluated
        :return: a pair of top-1 accuracy and weights memory size
        """
        device = self.queue.get()
        try:
            print(f"Quant config {quant_config['quant_conf']} is going to be evaluated on {device.name}")
            with tf.device(device.name):
                quantized_model = self.quantize_model_by_config(quant_config["quant_conf"])

                hardware_params = self._timeloop_pool.submit(self.eval_param, "hardware_params", quantized_model,
                                                             quant_config, device.name)

                accuracy = self.eval_param("accuracy", quantized_model, quant_config, device.name)

                # pool = ThreadPoolExecutor(max_workers=2)
                # (accuracy, hardware_params) = pool.map(
                #    lambda x: self.eval_param(x, quantized_model, quant_config, device.name),
                #    ["accuracy", "hardware_params"])

                return accuracy, hardware_params
        finally:
            self.queue.put(device)
