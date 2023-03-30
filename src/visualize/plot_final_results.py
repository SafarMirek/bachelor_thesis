import matplotlib.pyplot as plt
import glob
import json, gzip
from paretoarchive import PyBspTreeArchive
from matplotlib.ticker import FuncFormatter


def percent(x, pos=0):
    return '%.0f %%' % (100 * x)


memory_8bit_ref = 441736


def load_data(logs_dir, d_path, paretoFilter=False, paretoX="memory", paretoY="accuracy"):
    fn = sorted(glob.glob(logs_dir + "/%s/eval.*.json.gz" % d_path))
    if not fn:
        return []

    fn = fn[-1]
    print("# load %s" % fn)
    data = json.load(gzip.open(fn))

    # for i in range(len(data)):
    #    data[i]["memory"] = data[i]["power"] / ref_energy

    if paretoFilter:
        data = [data[x] for x in
                PyBspTreeArchive(2, minimizeObjective2=False).filter([(i[paretoX], i[paretoY]) for i in data],
                                                                     returnIds=True)]

    return sorted(data, key=lambda x: x[paretoX])


fig_def = {
    "title": "Mobilenet",
    "data": [
        {
            "source": "mobilenet_same",
            "color": "#fd8d3c",
            "marker": "o",
            "label": "stejná bitová přenost",
            "line_style": ":"
        }
    ]
}

fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
(ax1) = axes

ax1.set_title(fig_def["title"])

ax1.set_xlabel("Velikost vah v porovnání s 8-bitovým modelem [%]")
ax1.set_ylabel("Přesnost klasifikace [%]")

for data_def in fig_def["data"]:
    data = load_data("nsga_runs", data_def["source"], paretoFilter=True)
    # alldata += data

    ax1.step([x["memory"] / memory_8bit_ref for x in data], [x["accuracy"] for x in data], color=data_def["color"],
             label=data_def["label"], where="post", marker=data_def["marker"], markersize=5, linewidth=0.5,
             linestyle=data_def["line_style"])

ax1.set_ylim(0, 0.6)
ax1.set_xlim(0, 1)

ax1.xaxis.set_major_formatter(FuncFormatter(percent))
ax1.yaxis.set_major_formatter(FuncFormatter(percent))

ax1.legend()

fig.savefig("fig/final_results.png")
plt.show()
