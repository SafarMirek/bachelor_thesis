import glob
import gzip
import json
import re

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from paretoarchive import PyBspTreeArchive


def percent(x, pos=0):
    return '%.0f %%' % (100 * x)


memory_8bit_ref = 441736

uniform_data = [
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
        "accuracy": 0.5
    },
]

for record in uniform_data:
    record["memory_percent"] = record["memory"] / memory_8bit_ref

title = "Mobilenet"
alldata = {}
runs_folder = "mobilenet_qat_6"
for fn in sorted(glob.glob("nsga_runs/%s/run.*.json.gz" % runs_folder)):
    print(fn)

    gen = int(re.match(r".*run\.(\d+)\.json\.gz", fn).group(1))
    alldata[gen] = json.load(gzip.open(fn))
    for record in alldata[gen]["parent"] + alldata[gen]["offspring"]:
        record["memory_percent"] = record["memory"] / memory_8bit_ref

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
        oldp.set_offsets([[x["memory_percent"], x["accuracy"]] for x in oldpoints])
    # else:
    #    oldp.set_offsets([[x["energy"],x["accuracy"]] for x in data[k][:1]])

    datap.set_offsets([[x["memory_percent"], x["accuracy"]] for x in data[k]])

    pareto = PyBspTreeArchive(2, minimizeObjective2=False).filter(
        [[x["memory_percent"], x["accuracy"]] for x in data[k]])
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
oldp = plt.scatter([x["memory_percent"] for x in oldpoints], [x["accuracy"] for x in oldpoints], s=5,
                   color="tab:gray")
k = "parent"
datap = plt.scatter([x["memory_percent"] for x in data[k]], [x["accuracy"] for x in data[k]], s=5,
                    color="tab:red",
                    label="Best configurations (P)", zorder=100)
pareto = PyBspTreeArchive(2, minimizeObjective2=False).filter(
    [[x["memory_percent"], x["accuracy"]] for x in data[k]])
pareto = sorted(pareto)
dataline, = plt.step(*zip(*pareto), color="tab:red", linestyle=":", where="post")

# Tohle tady asi nedává v tomhle případě smysl, protože to je trénovaný jen na 6 epoch X 100 epoch pro eval
plt.scatter([x["memory_percent"] for x in uniform_data], [x["accuracy"] for x in uniform_data], s=10,
            color="tab:blue",
            marker="x",
            label="Uniform structure")

pareto = PyBspTreeArchive(2, minimizeObjective2=False).filter(
    [[x["memory_percent"], x["accuracy"]] for x in uniform_data])
pareto = sorted(pareto)
plt.step(*zip(*pareto), color="tab:blue", linestyle=":", where="post")

update(0)
anim = FuncAnimation(fig, update, frames=sorted(alldata.keys())[1:], interval=500)

plt.xlabel("Velikost vah v porovnání s 8-bitovým modelem [%]")
plt.ylabel("Přesnost klasifikace [%]")

ax.xaxis.set_major_formatter(FuncFormatter(percent))
ax.yaxis.set_major_formatter(FuncFormatter(percent))
ax.set_ylim(0, 0.6)
ax.set_xlim(0, 1)
# plt.scatter
plt.legend(loc="upper left")

anim.save("fig/nsga_generations.gif", writer='imagemagick', fps=1)
plt.show()
