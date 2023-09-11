import os
import re
import sys
import subprocess
import time
from datetime import datetime
import glob
import json
import csv
import shutil
import threading
import yaml
import multiprocessing
from typing import Tuple
from construct_workloads.create_model import create_keras_model
from construct_workloads.parse_model import parse_keras_model
from construct_workloads.construct_workloads import json_file_to_dict, construct_workloads


def dict_to_json(dictionary, filename):
    with open(filename, 'w') as json_file:
        json.dump(dictionary, json_file)


def extract_scalar_accesses_data(data, key):
    pattern = f"=== {key} ===\s*Total scalar accesses\s*:\s*([\d]+)\s*Op per Byte\s*:\s*([\d.]+)"
    match = re.search(pattern, data)
    if match:
        return {"Total scalar accesses": int(match.group(1)), "Op per Byte": float(match.group(2))}
    return None


def extract_memory_stats(architecture: str, data: str, result_dict: dict) -> dict:
        # Extract "Word bits"
        word_bits_pattern = "Word bits\s*:\s*(\d+)"
        word_bits_match = re.search(word_bits_pattern, data)
        if word_bits_match:
            word_bits = int(word_bits_match.group(1))
            result_dict["Word bits"] = word_bits

        # List of keys to extract data for
        if "simba" in architecture:
            keys_to_extract = ["PEWeightRegs", "PEAccuBuffer", "PEWeightBuffer", "PEInputBuffer", "GlobalBuffer", "DRAM"]
        else: # assume eyeriss
            keys_to_extract = ["psum_spad", "weights_spad", "ifmap_spad", "shared_glb", "DRAM"]

        # Extract data for each key and add it to the result_dict
        for key in keys_to_extract:
            extracted_data = extract_scalar_accesses_data(data, key)
            if extracted_data:
                result_dict[f"{key} data"] = extracted_data
        return result_dict


class MapperFacade:
    """Class represents the facade interface for calling timeloop mapper and retrieve hardware metrics.

    The __init__ method of this class takes in the relative path to the timeloop configs folder
    and the name of the HW architecture to be used for invoking the subsequent timeloop-mapper calls.

    Args:
        configs_rel_path (str): Relative path to the timeloop configs folder.
        architecture (str): Name of the architecture to be used along with its associated components and constraints.
    """
    def __init__(self, configs_rel_path: str = "timeloop_configs", architecture: str = "eyeriss") -> None:
        self._architecture = architecture
        self._mode = f"timeloop-mapper"
        self._thread_id = threading.get_ident()

        self.configs_path = configs_rel_path
        self.arch = glob.glob(f"{self.configs_path}/architectures/{architecture}/*.yaml")[0]
        self.components = glob.glob(f"{self.configs_path}/architectures/{architecture}/components/*.yaml")
        self.constraints = glob.glob(f"{self.configs_path}/architectures/{architecture}/constraints/*.yaml")

    """Method to modify the mapper heuristic settings for the given settings.
    
    Args:
        mapper_config (dict): Parsed yaml config containing the mapper heuristic settings.
        heuristic (str): Name of the mapper heuristic to be used. Choices are `exhaustive`, `hybrid`, `linear` or `random`.
        threads (str): Number of threads to be used by the mapper heuristics. Choices are `all` or integer number.
        total_valid (int, optional): Specifies the number of valid mappings to be considered by the mapper heuristic. A value of 0 means that this criteria is not used for thread termination.
        Lower values will result in a reduced search space proportional to the workload`s dimensionality. Defaults to 0.
        log_all (bool): Flag to log all the mappings.
    Returns:
        dict: Modified mapper heuristic settings.    
    """
    def _modify_mapper_configs(self, mapper_config, heuristic, metrics, threads, total_valid, log_all):
        mapper_config["mapper"]["out_prefix"] = f"{self._mode}_{self._thread_id}"
        mapper_config["mapper"]["optimization-metrics"] = list(metrics) if metrics[1] else [metrics[0]]
        mapper_config["mapper"]["search-size"] = int(total_valid / threads) if threads != "all" else int(total_valid / multiprocessing.cpu_count())
        
        if log_all:
            mapper_config["mapper"]["log-oaves"] = True
            mapper_config["mapper"]["log-suboptimal"] = True
            mapper_config["mapper"]["log-all"] = True
            mapper_config["mapper"]["log-stats"] = True
        
        if isinstance(threads, int):
            mapper_config["mapper"]["num-threads"] = threads
        
        if heuristic == "exhaustive":
            mapper_config["mapper"]["algorithm"] = "linear-pruned" 
            mapper_config["mapper"]["victory-condition"] = 0
            mapper_config["mapper"]["timeout"] = 0
            # Remove the "max-permutations-per-if-visit" if it exists
            mapper_config["mapper"].pop("max-permutations-per-if-visit", None)
        else:
            if total_valid == 0:
                mapper_config["mapper"]["victory-condition"] = 500
                mapper_config["mapper"]["max-permutations-per-if-visit"] = 16
                mapper_config["mapper"]["timeout"] = 15000
            else:
                mapper_config["mapper"].pop("victory-condition", None)
                mapper_config["mapper"].pop("max-permutations-per-if-visit", None)
                mapper_config["mapper"].pop("timeout", None)

            if heuristic == "random":
                mapper_config["mapper"]["algorithm"] = "random-pruned"
            elif heuristic == "linear":
                mapper_config["mapper"]["algorithm"] = "linear-pruned"
            elif heuristic == "hybrid":
                mapper_config["mapper"]["algorithm"] = "hybrid"

        return mapper_config

    """Method to run timeloop-mapper for a given workload (i.e. a CNN layer) and mapper heuristic settings.

    Args:
        workload (str): Relative path to the workload config file.
        threads (object, optional): Number of threads to be used by the mapper heuristics. Choices are `all` or integer number. Defaults to "all".
        heuristic (str, optional): Name of the mapper heuristic to be used. Choices are `exhaustive`, `hybrid`, `linear` or `random`. Defaults to "random".
        metrics (tuple, optional): Tuple of two metrics to be used for the mapper heuristic. Possible values are all six combinations of `energy`, `delay`, `lla`
        with an additional seventh option `edp`, leaving the second metric blank. Defaults to ("energy", "delay").
        total_valid (int, optional): Specifies the number of valid mappings to be considered by the mapper heuristic. A value of 0 means that this criteria is not used for thread termination.
        Lower values will result in a reduced search space proportional to the workload`s dimensionality. Defaults to 0.
        out_dir (str, optional): Relative path to the output directory where the timeloop-mapper output files are stored. Defaults to "tmp_outputs".
        log_all (bool, optional): Flag to log all the mappings. Defaults to False.
        verbose (bool, optional): Flag to print the timeloop-mapper output. Defaults to False.
        clean (bool, optional): Flag to delete the temporary files generated by timeloop-mapper. Defaults to True.

    Returns:
        dict: Dictionary containing the best mapping's HW parameters and total runtime of the timeloop-mapper call.
    """
    def run_one_workload(self, workload: str, bitwidth: str, batch_size: int = 1, threads: object = "all", heuristic: str = "random", metrics: Tuple[str, str] = ("energy", "delay"), total_valid: int = 0, out_dir: str = "tmp_outputs", log_all: bool = False, verbose: bool = False, clean: bool = True) -> dict:
        mapper = f"{self.configs_path}/mapper_heuristics/mapper.yaml"
        cache_dir = "timeloop_cache"
        cache_name = f"cache_{self._architecture}.json"

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if os.path.exists(f"{cache_dir}/{cache_name}"):
            with open(f"{cache_dir}/{cache_name}", "r") as file:
                cache = json.load(file)
        else:
            cache = {}

        layer = workload.split("/")[-1].split(".")[0]
        if layer in cache:
            if bitwidth in cache[layer]:
                # Return dictionary with the best found HW params and total mapper runtime from cache
                return cache[layer][bitwidth]
        else:
            cache[layer] = {}  # Initialize cache[layer] as a dictionary
 
        with open(mapper, "r") as map:
            try:
                config_dict = yaml.safe_load(map)
            except yaml.YAMLError as e:
                print(e)
                sys.exit(1)

        # Modify the mapper heuristic settings for the given settings
        config_dict = self._modify_mapper_configs(config_dict, heuristic, metrics, threads, total_valid, log_all)

        # Write the modified YAML data to a temporary file
        modified_mapper = os.path.splitext(mapper)[0] + f"_{self._thread_id}.yaml"
        with open(modified_mapper, "w") as modified_map:
            yaml.dump(config_dict, modified_map)

        start_time = time.time()
        tmp_dir = f"{out_dir}_{self._thread_id}"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        # Running the timeloop-mapper for the given workload and chosen mapper heuristic settings
        if verbose:
            subprocess.run([self._mode, self.arch] + self.components + self.constraints
                        + [modified_mapper, workload, "-o", tmp_dir], check=True)
        else:
            subprocess.run([self._mode, self.arch] + self.components + self.constraints
                        + [modified_mapper, workload, "-o", tmp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Reading the CSV file into a dictionary
        with open(f"{tmp_dir}/{self._mode}_{self._thread_id}.stats.csv", "r") as f:
            reader = csv.DictReader(f)
            result_dict = next(reader)
        
        # Read the content of the text file to retrieve the total scalar accesses and Op per Byte
        with open(f"{tmp_dir}/{self._mode}_{self._thread_id}.stats.txt", "r") as f:
            data = f.read()
            # Add the total scalar accesses and Op per Byte to the result dictionary
            result_dict = extract_memory_stats(self._architecture, data, result_dict)

        # Deleting the tmp files
        if clean:
            shutil.rmtree(tmp_dir)
            os.remove(modified_mapper)

        end_time = time.time()
        runtime = end_time - start_time

        cache[layer][bitwidth] = {"Mode": self._mode, "HW": self._architecture, "Workload": layer, "Bitwidths": bitwidth, "Batch_size": batch_size, "Mapper heuristic": heuristic, "Total valid": total_valid, "Threads": threads, "Optimized_metric_1": metrics[0], "Optimized_metric_2": metrics[1], **result_dict, "Runtime [s]": "{:.2f}".format(runtime)}
        with open(f"{cache_dir}/{cache_name}", "w") as file:
            json.dump(cache, file, indent=2)

        # Return dictionary with the best found HW params and total mapper runtime
        return cache[layer][bitwidth]


    """Method to run timeloop-mapper for all workloads (i.e. a CNN network's layers) in a given folder and mapper heuristic settings.

    Args:
        workload (str): Relative path to the workload config file.
        batch_size (int, optional): Batch size to be used for the model within the timeloop mapper. Defaults to 1.
        bitwidths (object, optional): Bitwidths setting to be used for the model's workloads within timeloop mapper. Choices are:
                                    None, tuple (i.e. (8,4,8)), dict representing non-uniform bitwidths for each layer,
                                    (for example: `{"layer_1": {"Inputs": 8, "Weights": 4, "Outputs": 8},
                                    "layer_2": {"Inputs": 5, "Weights": 2, "Outputs": 3}}`)
                                    Defaults to None.
        threads (object, optional): Number of threads to be used by the mapper heuristics. Choices are `all` or integer number. Defaults to "all".
        heuristic (str, optional): Name of the mapper heuristic to be used. Choices are `exhaustive`, `hybrid`, `linear` or `random`. Defaults to "random".
        metrics (tuple, optional): Tuple of two metrics to be used for the mapper heuristic. Possible values are all six combinations of `energy`, `delay`, `lla`
        with an additional seventh option `edp`, leaving the second metric blank. Defaults to ("energy", "delay").
        total_valid (int, optional): Specifies the number of valid mappings to be considered by the mapper heuristic. A value of 0 means that this criteria is not used for thread termination.
        Lower values will result in a reduced search space proportional to the workload`s dimensionality. Defaults to 0.
        out_dir (str, optional): Relative path to the output directory where the timeloop-mapper output files are stored. Defaults to "tmp_outputs".
        log_all (bool, optional): Flag to log all the mappings. Defaults to False.
        verbose (bool, optional): Flag to print the timeloop-mapper output. Defaults to False.
        clean (bool, optional): Flag to delete the temporary files generated by timeloop-mapper. Defaults to True.

    Returns:
        dict: Dictionary containing the best mappings HW parameters and total runtime of the individual workloads timeloop-mapper calls.
    """
    def run_all_workloads(self, workloads: str, batch_size: int = 1, bitwidths: object = None, threads: object = "all", heuristic: str = "random", metrics: tuple = ("energy", "delay"), total_valid: int = 0, out_dir: str = "tmp_outputs", log_all: bool = False, verbose: bool = False, clean: bool = True) -> dict:
        workloads = glob.glob(f"{workloads}/*.yaml")
        hw_params = {}


        # Retrieve parameters for each workload
        for i, workload in enumerate(workloads):
            if bitwidths is None:
                bitwidth = "native_native_native"
            elif isinstance(bitwidths, tuple):
                bitwidth = f"{bitwidths[0]}_{bitwidths[1]}_{bitwidths[2]}"
            else:
                key = list(bitwidths.keys())[i]
                bitwidth = f"{bitwidths[key]['Inputs']}_{bitwidths[key]['Weights']}_{bitwidths[key]['Outputs']}"
            hw_params[workload.split("/")[-1].split(".")[0]] = self.run_one_workload(workload=workload, batch_size=batch_size, bitwidth=bitwidth, threads=threads, heuristic=heuristic, metrics=metrics, total_valid=total_valid, out_dir=f"{out_dir}/{workload.split('/')[-1].split('.')[0]}", log_all=log_all, verbose=verbose, clean=clean)
            print("Finished workload ", i+1, "/", len(workloads))

        # Return dictionary with individual workload's HW params and runtime
        return hw_params

    """Method to create a cnn model and run timeloop-mapper for all workloads (i.e. a CNN network's layers) for given configuration and mapper heuristic settings.

    Args:
        api_choice (str): Name of the API to be used. Choices are `keras` or `pytorch`.
        model (str): Model from torchvision choices:
                    `resnet18`, `alexnet`, `vgg16`, `squeezenet`, `densenet`,
                    `inception_v3`, `googlenet`, `shufflenet`
                    `mobilenet_v2`, `wide_resnet50_2`, `mnasnet`

                    model from tensorflow.keras.applications choices:
                    `xception`, `vgg16`, `vgg19`, `resnet50`, `resnet101`,
                    `resnet152`, `resnet50_v2`, `resnet101_v2`, `resnet152_v2`,
                    `inception_v3`, `inception_resnet_v2`, `mobilenet`, `mobilenet_v2`,
                    `densenet121`, `densenet169`, `densenet201`, `nasnet_large`,
                    `nasnet_mobile`
        batch_size (int, optional): Batch size to be used for the model within the timeloop mapper. Defaults to 1.
        bitwidths (object, optional): Bitwidths setting to be used for the model's workloads within timeloop mapper. Choices are:
                                    None, tuple (i.e. (8,4,8)), dict representing non-uniform bitwidths for each layer,
                                    (for example: `{"layer_1": {"Inputs": 8, "Weights": 4, "Outputs": 8},
                                    "layer_2": {"Inputs": 5, "Weights": 2, "Outputs": 3}}`)
                                    Defaults to None.
        input_size (str, optional): Input size of the model. Defaults to "224,224,3".
        threads (object, optional): Number of threads to be used by the mapper heuristics. Choices are `all` or integer number. Defaults to "all".
        heuristic (str, optional): Name of the mapper heuristic to be used. Choices are `exhaustive`, `hybrid`, `linear` or `random`. Defaults to "random".
        metrics (tuple, optional): Tuple of two metrics to be used for the mapper heuristic. Possible values are all six combinations of `energy`, `delay`, `lla`
        with an additional seventh option `edp`, leaving the second metric blank. Defaults to ("energy", "delay").
        total_valid (int, optional): Specifies the number of valid mappings to be considered by the mapper heuristic. A value of 0 means that this criteria is not used for thread termination.
        Lower values will result in a reduced search space proportional to the workload`s dimensionality. Defaults to 0.
        out_dir (str, optional): Relative path to the output directory where the timeloop-mapper output files are stored. Defaults to "tmp_outputs".
        log_all (bool, optional): Flag to log all the mappings. Defaults to False.
        verbose (bool, optional): Flag to print the timeloop-mapper output. Defaults to False.
        clean (bool, optional): Flag to delete the temporary files generated by timeloop-mapper. Defaults to True.

    Returns:
        dict: Dictionary containing the best mappings HW parameters and total runtime of the individual workloads of the given cnn model.
    """
    def get_hw_params_create_model(self, api_choice: str, model: str, batch_size: int = 1, bitwidths: object = None, input_size: str = "224,224,3", threads: object = "all", heuristic: str = "random", metrics: tuple = ("energy", "delay"), total_valid: int = 0, out_dir: str = "tmp_outputs", log_all: bool = False, verbose: bool = False, clean: bool = True) -> dict:
        if isinstance(input_size, str):
            input_size = tuple((int(d) for d in str.split(input_size, ",")))
        # Create templates for individual model's CONV layers
        if api_choice == "keras":
            create_keras_model(api_name=api_choice, model_name=model, input_size=input_size, batch_size=batch_size, out_dir="construct_workloads/parsed_models", out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        elif api_choice == "pytorch":
            pass
            #create_pytorch_model(api_name=api_choice, model_name=model, input_size=input_size, batch_size=batch_size, out_dir="construct_workloads/parsed_models", out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        else:
            raise ValueError("Invalid API choice. Choose between `keras` or `pytorch`.")

        # Construct timeloop workloads from the created templates and add to them the bitwidth settings
        yaml_model = f"construct_workloads/parsed_models/{model.split('/')[-1].split('.')[0]}.yaml"
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        workloads_location = f"{self.configs_path}/workload_shapes/{timestamp}"

        if bitwidths is None:
            construct_workloads(model=yaml_model, bitwidth_setting="native", uniform_width_set=None, non_uniform_width_set=None, out_dir=workloads_location, out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        elif isinstance(bitwidths, dict):
            construct_workloads(model=yaml_model, bitwidth_setting="non-uniform", uniform_width_set=None, non_uniform_width_set=bitwidths, out_dir=workloads_location, out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        elif isinstance(bitwidths, tuple) and len(bitwidths) == 3 and all(isinstance(item, int) for item in bitwidths):
            construct_workloads(model=yaml_model, bitwidth_setting="uniform", uniform_width_set=bitwidths, non_uniform_width_set=None, out_dir=workloads_location, out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        else:
            print("Unrecognized bitwidths object. Expected dict, tuple or None to represent non-uniform, uniform and native bitwidhts settings, respectively.", file=sys.stderr)
            sys.exit(0)

        # Run timeloop-mapper on the created workloads
        results = self.run_all_workloads(workloads=workloads_location, batch_size=batch_size, bitwidths=bitwidths, threads=threads, heuristic=heuristic, metrics=metrics, total_valid=total_valid, out_dir=out_dir, log_all=log_all, verbose=verbose, clean=clean)
        return results

    """Method to parse a cnn model and run timeloop-mapper for all workloads (i.e. a CNN network's layers) for given configuration and mapper heuristic settings.

    Args:
        model (str): Path to the cnn model to be parsed.
        batch_size (int, optional): Batch size to be used for the model within the timeloop mapper. Defaults to 1.
        bitwidths (object, optional): Bitwidths setting to be used for the model's workloads within timeloop mapper. Choices are:
                                    None, tuple (i.e. (8,4,8)), dict representing non-uniform bitwidths for each layer,
                                    (for example: `{"layer_1": {"Inputs": 8, "Weights": 4, "Outputs": 8},
                                    "layer_2": {"Inputs": 5, "Weights": 2, "Outputs": 3}}`)
                                    Defaults to None.
        input_size (str, optional): Input size of the model. Defaults to "224,224,3".
        threads (object, optional): Number of threads to be used by the mapper heuristics. Choices are `all` or integer number. Defaults to "all".
        heuristic (str, optional): Name of the mapper heuristic to be used. Choices are `exhaustive`, `hybrid`, `linear` or `random`. Defaults to "random".
        metrics (tuple, optional): Tuple of two metrics to be used for the mapper heuristic. Possible values are all six combinations of `energy`, `delay`, `lla`
        with an additional seventh option `edp`, leaving the second metric blank. Defaults to ("energy", "delay").
        total_valid (int, optional): Specifies the number of valid mappings to be considered by the mapper heuristic. A value of 0 means that this criteria is not used for thread termination.
        Lower values will result in a reduced search space proportional to the workload`s dimensionality. Defaults to 0.
        out_dir (str, optional): Relative path to the output directory where the timeloop-mapper output files are stored. Defaults to "tmp_outputs".
        log_all (bool, optional): Flag to log all the mappings. Defaults to False.
        verbose (bool, optional): Flag to print the timeloop-mapper output. Defaults to False.
        clean (bool, optional): Flag to delete the temporary files generated by timeloop-mapper. Defaults to True.

    Returns:
        dict: Dictionary containing the best mappings HW parameters and total runtime of the individual workloads of the given cnn model.
    """
    def get_hw_params_parse_model(self, model: str, batch_size: int = 1, bitwidths: object = None, input_size: str = "224,224,3", threads: object = "all", heuristic: str = "random", metrics: tuple = ("energy", "delay"), total_valid: int = 0, out_dir: str = "tmp_outputs", log_all: bool = False, verbose: bool = False, clean: bool = True) -> dict:
        if not os.path.exists(model):
            raise FileNotFoundError(f"No model file '{model}' found.")

        # Deduct the API from the model file extension
        if model.split("/")[-1].split(".")[-1] == "h5" or model.split("/")[-1].split(".")[-1] == "keras":
            api_choice = "keras"
        elif model.split("/")[-1].split(".")[-1] == "pt" or model.split("/")[-1].split(".")[-1] == "pth":
            api_choice = "pytorch"
        else:
            print("Unrecognized model file extension. Expected .keras, .h5 for keras OR .pt, .pth for pytorch.", file=sys.stderr)
            sys.exit(0)

        # Create templates for individual model's CONV layers
        if api_choice == "keras":
            parse_keras_model(api_name=api_choice, model_file=model, input_size=input_size, batch_size=batch_size, out_dir="construct_workloads/parsed_models", out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        #else:
            #parse_pytorch_model(api_name=api_choice, model_file=model, input_size=input_size, batch_size=batch_size, out_dir="construct_workloads/parsed_models", out_file=model.split("/")[-1].split(".")[0], verbose=verbose)

        # Construct timeloop workloads from the created templates and add to them the bitwidth settings
        yaml_model = f"construct_workloads/parsed_models/{model.split('/')[-1].split('.')[0]}.yaml"
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        workloads_location = f"{self.configs_path}/workload_shapes/{timestamp}"

        if bitwidths is None:
            construct_workloads(model=yaml_model, bitwidth_setting="native", uniform_width_set=None, non_uniform_width_set=None, out_dir=workloads_location, out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        elif isinstance(bitwidths, dict):
            construct_workloads(model=yaml_model, bitwidth_setting="non-uniform", uniform_width_set=None, non_uniform_width_set=bitwidths, out_dir=workloads_location, out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        elif isinstance(bitwidths, tuple) and len(bitwidths) == 3 and all(isinstance(item, int) for item in bitwidths):
            construct_workloads(model=yaml_model, bitwidth_setting="uniform", uniform_width_set=bitwidths, non_uniform_width_set=None, out_dir=workloads_location, out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        else:
            print("Unrecognized bitwidths object. Expected dict, tuple or None to represent non-uniform, uniform and native bitwidhts settings, respectively.", file=sys.stderr)
            sys.exit(0)

        # Run timeloop-mapper on the created workloads
        results = self.run_all_workloads(workloads=workloads_location, batch_size=batch_size, bitwidths=bitwidths, threads=threads, heuristic=heuristic, metrics=metrics, total_valid=total_valid, out_dir=out_dir, log_all=log_all, verbose=verbose, clean=clean)
        return results


if __name__ == "__main__":
    facade = MapperFacade()
    # Example usage run creating and evaluating workloads for alexnet pytorch model with no quantization (bitwidths=None)
    # results = facade.get_hw_params_create_model(api_choice="pytorch", model="alexnet", batch_size=1, bitwidths=None, input_size="224,224,3", threads="all", heuristic="random", metrics=("energy", "delay"), clean=True)
    # dict_to_json(results, "results_native.json")

    # Example usage run creating and evaluating workloads for alexnet pytorch model with uniform quantization for each layer (bitwidths=(8,4,8))
    # results = facade.get_hw_params_create_model(api_choice="pytorch", model="alexnet", batch_size=1, bitwidths=(8,4,8), input_size="224,224,3", threads="all", heuristic="random", metrics=("energy", "delay"), clean=True)
    # dict_to_json(results, "results_uniform.json")

    # Example usage run creating and evaluating workloads for parsed mobilenet keras model with non-uniform quantization for each layer (bitwidths=dict)
    json_dict = json_file_to_dict("construct_workloads/temps/bitwidths_mobilenet_sample.json")
    results = facade.get_hw_params_parse_model(model="mobilenet_tinyimagenet_025.keras", batch_size=1, bitwidths=json_dict, input_size="224,224,3", threads="all", heuristic="random", metrics=("edp", ""), verbose=True)
    dict_to_json(results, "results_non_uniform.json")
