# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import argparse
import glob
import gzip
import json
import re

import inquirer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from paretoarchive import PyBspTreeArchive


def percent(x, pos=0):
    return '%.0f %%' % (100 * x)


parser = argparse.ArgumentParser(
    prog='plot_nsga_generations',
    description='Plot NSGA generations',
    epilog='')

parser.add_argument('--approx', default=False, action='store_true')
parser.add_argument('--per-channel', default=False, action='store_true')
parser.add_argument('--symmetric', default=False, action='store_true')
parser.add_argument('--configurations', default="visualize/configurations.json")
parser.add_argument('--uniform-data', default="visualize/uniform_data.json")
parser.add_argument('--act-quant', default=False, action='store_true')
parser.add_argument('--regex', default=None, type=str)

args = parser.parse_args()

uniform_data = json.load(open(args.uniform_data))

loaded_configurations = json.load(open(args.configurations))
configurations = list(
    filter(
        lambda x: x["per_channel"] == args.per_channel and x["symmetric"] == args.symmetric and x[
            "approx"] == args.approx and x["act_quant"] == args.act_quant and (
                          args.regex is None or re.match(args.regex, x["run_folder"])),
        loaded_configurations
    )
)

if len(configurations) != 1:
    if len(configurations) == 0:
        print("No configuration for this setting was not found. Available configurations:")
        configurations = loaded_configurations
    questions = [
        inquirer.List('configuration',
                      message="Select configuration",
                      choices=[cfg["run_folder"] for cfg in configurations],
                      ),
    ]
    answers = inquirer.prompt(questions)
    print(answers)
    configuration = next(filter(lambda x: x["run_folder"] == answers["configuration"], configurations))
else:
    configuration = configurations[0]

memory_8bit_ref = configuration["base_memory"]
ref_accuracy = configuration["float_accuracy"]
title = configuration["title"]
alldata = {}
runs_folder = configuration["run_folder"]
for fn in sorted(glob.glob("nsga_runs/%s/run.*.json.gz" % runs_folder)):
    print(fn)

    gen = int(re.match(r".*run\.(\d+)\.json\.gz", fn).group(1))
    alldata[gen] = json.load(gzip.open(fn))
    for record in alldata[gen]["parent"] + alldata[gen]["offspring"]:
        record["memory_percent"] = record["memory"] / memory_8bit_ref
        record["accuracy_percent"] = record["accuracy"] / ref_accuracy

datap = None
oldp = None
oldpoints = []


def update(i):
    global datap, oldp, oldpoints, dataline
    gen = i

    if not i:
        oldpoints = []
    data = alldata[gen]
    plt.title("%s (gen = %d)" % (title, gen))
    k = "parent"

    if oldpoints:
        oldp.set_offsets([[x["memory_percent"], x["accuracy_percent"]] for x in oldpoints])
    # else:
    #    oldp.set_offsets([[x["energy"],x["accuracy"]] for x in data[k][:1]])

    datap.set_offsets([[x["memory_percent"], x["accuracy_percent"]] for x in data[k]])

    pareto = PyBspTreeArchive(2, minimizeObjective2=False).filter(
        [[x["memory_percent"], x["accuracy_percent"]] for x in data[k]])
    pareto = sorted(pareto)
    dataline.set_data(*zip(*pareto))

    oldpoints = data[k]
    return datap, oldp


gen = 0
fig = plt.figure()
gen = 0
ax = plt.gca()

data = alldata[gen]
plt.title("%s (gen = %d)" % (title, gen))
oldp = plt.scatter([x["memory_percent"] for x in oldpoints], [x["accuracy_percent"] for x in oldpoints], s=5,
                   color="tab:gray")
k = "parent"
datap = plt.scatter([x["memory_percent"] for x in data[k]], [x["accuracy_percent"] for x in data[k]], s=5,
                    color="tab:red",
                    label="Best configurations (P)", zorder=100)
pareto = PyBspTreeArchive(2, minimizeObjective2=False).filter(
    [[x["memory_percent"], x["accuracy_percent"]] for x in data[k]])
pareto = sorted(pareto)
dataline, = plt.step(*zip(*pareto), color="tab:red", linestyle=":", where="post")

# Tohle tady asi nedává v tomhle případě smysl, protože to je trénovaný jen na 6 epoch X 100 epoch pro eval
# plt.scatter([x["memory_percent"] for x in uniform_data], [x["accuracy"] for x in uniform_data], s=10,
#            color="tab:blue",
#            marker="x",
#            label="Uniform structure")

# pareto = PyBspTreeArchive(2, minimizeObjective2=False).filter(
#    [[x["memory_percent"], x["accuracy"]] for x in uniform_data])
# pareto = sorted(pareto)
# plt.step(*zip(*pareto), color="tab:blue", linestyle=":", where="post")

update(0)
anim = FuncAnimation(fig, update, frames=sorted(alldata.keys())[1:], interval=500)

plt.xlabel("Velikost vah v porovnání s 8-bitovým modelem [%]")
plt.ylabel("Top-1 relativní přesnost po částečném dotrénování [%]")

ax.xaxis.set_major_formatter(FuncFormatter(percent))
ax.yaxis.set_major_formatter(FuncFormatter(percent))
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 1)
# plt.scatter
plt.legend(loc="upper left")

anim.save("fig/nsga_generations.gif", writer='imagemagick', fps=1)
plt.show()
