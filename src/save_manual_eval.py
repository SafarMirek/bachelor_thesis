import json
import gzip
import argparse

# Script arguments
parser = argparse.ArgumentParser(
    prog='save_manual_eval',
    description='Creates default output file of nsga eval from manually obtained data',
    epilog='')

parser.add_argument('--output-file', '-o', required=True, type=str)

args = parser.parse_args()

data = [
    {
        "quant_conf": [2 for _ in range(28)],
        "memory": 167266,
        "accuracy": 0.0524
    },
    {
        "quant_conf": [3 for _ in range(28)],
        "memory": 213011,
        "accuracy": 0.3998
    },
    {
        "quant_conf": [4 for _ in range(28)],
        "memory": 258756,
        "accuracy": 0.4246
    },
    {
        "quant_conf": [5 for _ in range(28)],
        "memory": 304501,
        "accuracy": 0.4692
    },
    {
        "quant_conf": [6 for _ in range(28)],
        "memory": 350246,
        "accuracy": 0.4938
    },
    {
        "quant_conf": [7 for _ in range(28)],
        "memory": 395991,
        "accuracy": 0.4978
    },
    {
        "quant_conf": [8 for _ in range(28)],
        "memory": 441736,
        "accuracy": 0.5076
    },
]

json.dump(data, gzip.open(args.output_file, "wt", encoding="utf8"))
