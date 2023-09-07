# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf

import mobilenet_tinyimagenet_qat
import nsga.nsga_qat
from nsga.nsga import NSGAAnalyzer

from queue import Queue

from mapper_facade import MapperFacade


class MultiGPUQATNSGA(nsga.nsga_qat.QATNSGA):
    """
    NSGA-II for proposed system, it uses MultiGPU QAT Analyzer for evaluation of individuals
    """

    def __init__(self, logs_dir, base_model_path, parent_size=50, offspring_size=50, generations=25, batch_size=128,
                 qat_epochs=10, previous_run=None, cache_datasets=False, approx=False, activation_quant_wait=0,
                 per_channel=True, symmetric=True, learning_rate=0.2, timeloop_heuristic="random"):
        super().__init__(logs_dir, base_model_path, parent_size, offspring_size, generations, batch_size, qat_epochs,
                         previous_run, cache_datasets, approx, activation_quant_wait, per_channel, symmetric,
                         learning_rate, timeloop_heuristic)

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
                                   base_model_path=self.base_model_path)


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
                 logs_dir_pattern=None, checkpoints_dir_pattern=None, timeloop_heuristic="random"):
        super().__init__(base_model_path, batch_size, qat_epochs, bn_freeze, learning_rate, warmup, cache_datasets,
                         approx,
                         activation_quant_wait, per_channel, symmetric, logs_dir_pattern, checkpoints_dir_pattern,
                         timeloop_heuristic)
        self._queue = None
        self._timeloop_pool = ThreadPoolExecutor(max_workers=1)

    @property
    def queue(self):
        if self._queue is None:
            self._queue = Queue()
            logical_devices = tf.config.list_logical_devices('GPU')
            for device in logical_devices:
                self._queue.put(device)
        return self._queue

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

            if not any(x["quant_conf"] == quant_conf for x in self.cache):  # not in cache
                if not any(conf == quant_conf for conf in needs_eval):  # not in needs eval
                    needs_eval.append(quant_conf)

        # Run evaluation for all configurations that needs it
        logical_devices = tf.config.list_logical_devices('GPU')
        pool = ThreadPoolExecutor(max_workers=len(logical_devices))  # One thread for timeloop

        print("Needs eval: " + str(needs_eval) + " on " + str(len(logical_devices)) + " GPUs.")
        results = list(pool.map(self.get_eval_of_config, needs_eval))
        print("Eval done")

        for i, quant_conf in enumerate(needs_eval):
            node = {
                "quant_conf": quant_conf,
                "accuracy": float(results[i][0]),
                "total_edp": float(results[i][1].result()["total_edp"]),
                #"total_energy": float(results[i][1].result()["total_energy"]),
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
                total_edp = cache_sel[0]["total_edp"]
                #total_cycles = cache_sel[0]["total_cycles"]
                tf.print("Cache : %s;accuracy=%s;edp=%s;" % (
                    str(quant_conf), accuracy, total_edp))
            else:  # Not found in cache
                raise ValueError("All configurations should be in cache, how has this happends?")

            # Create output node
            node = node_conf.copy()
            node["quant_conf"] = quant_conf
            node["accuracy"] = float(accuracy)
            node["total_edp"] = float(total_edp)
            #node["total_cycles"] = int(total_cycles)

            yield node

    def eval_param(self, eval_param, quantized_model, quant_config, device_name):
        if eval_param == "hardware_params":
            mapper_facade = MapperFacade()
            total_valid = 0 if self.timeloop_heuristic == "exhaustive" else 30000
            hardware_params = mapper_facade.get_hw_params_parse_model(model=self.base_model_path, batch_size=1,
                                                                      bitwidths=nsga.nsga_qat.get_config_from_model(
                                                                          quantized_model),
                                                                      input_size="224,224,3", threads=24,
                                                                      heuristic=self.timeloop_heuristic,
                                                                      metrics=("edp", ""),
                                                                      total_valid=total_valid,
                                                                      verbose=True)
            total_edp = sum(map(lambda x: float(x["EDP [J*cycle]"]), hardware_params.values()))
            #total_cycles = sum(map(lambda x: int(x["Cycles"]), hardware_params.values()))

            return {"total_edp": total_edp}

        elif eval_param == "accuracy":
            with tf.device(device_name):
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
                return accuracy

    def get_eval_of_config(self, quant_config):
        """
        Get evaluation for configuration
        :param quant_config:  to be evaluated
        :return: a pair of top-1 accuracy and weights memory size
        """
        device = self.queue.get()
        try:
            print(f"Quant config {quant_config} is going to be evaluated on {device.name}")
            with tf.device(device.name):
                quantized_model = self.quantize_model_by_config(quant_config)

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
