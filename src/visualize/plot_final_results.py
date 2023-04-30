import matplotlib.pyplot as plt
import glob
import json, gzip
from paretoarchive import PyBspTreeArchive
from matplotlib.ticker import FuncFormatter


def percent(x, pos=0):
    return '%.0f %%' % (100 * x)


def load_data(logs_dir, d_path, paretoFilter=False, paretoX="memory", paretoY="accuracy"):
    fn = sorted(glob.glob(logs_dir + "/%s/eval.*.json.gz" % d_path))
    if not fn:
        return []

    fn = fn[-1]
    print("# load %s" % fn)
    data = json.load(gzip.open(fn))

    orig_fn = fn.replace("eval.", "")
    try:
        orig_data = json.load(gzip.open(orig_fn))
        orig_data = orig_data["parent"] + orig_data["offspring"]
    except:
        orig_data = []

    # for i in range(len(data)):
    #    data[i]["memory"] = data[i]["power"] / ref_energy

    if paretoFilter:
        data = [data[x] for x in
                PyBspTreeArchive(2, minimizeObjective2=False).filter([(i[paretoX], i[paretoY]) for i in data],
                                                                     returnIds=True)]

    for record in data:
        before_finetuning = list(filter(lambda x: x["quant_conf"] == record["quant_conf"], orig_data))
        if len(before_finetuning) > 0:
            record["accuracy_before"] = before_finetuning[0]["accuracy"]
            record["accuracy_max"] = max(record["accuracy"], record["accuracy_before"])
        else:
            record["accuracy_max"] = record["accuracy"]

    return sorted(data, key=lambda x: x[paretoX])


figure_definitions = [
    {
        "title": "Mobilenet 0.25 (Per-layer asymetrická kvantizace)",
        "title_en": "Mobilenet 0.25 (Per-layer asymmetric quantization)",
        "output_file": "final_results_per_layer_asymmetric_025",
        "base_memory": 1957472,
        "base_accuracy": 0.516,
        "data": [
            {
                "source": "mobilenet_025_qat_12_no_act_approx_per_layer_asymmetric_24pch",
                "color": "tab:red",
                "marker": "o",
                "label": "Navržený systém",
                "label_en": "Proposed system",
                "line_style": ":"
            },
            {
                "source": "mobilenet_025_same_per_layer_asymmetric",
                "color": "black",
                "marker": "x",
                "label": "stejná bitová přenost",
                "label_en": "uniform bit precision",
                "line_style": ":"
            },
        ],
    },
    {
        "title": "Mobilenet 0.25 (Per-layer asymetrická kvantizace)",
        "title_en": "Mobilenet 0.25 (Per-layer asymmetric quantization)",
        "output_file": "final_results_per_layer_asymmetric_025_accurate_eval",
        "base_memory": 1957472,
        "base_accuracy": 0.516,
        "data": [
            {
                "source": "mobilenet_025_qat_12_no_act_approx_per_layer_asymmetric_24pch_accurate_eval",
                "color": "tab:cyan",
                "marker": "s",
                "label": "Navržený systém (accurate)",
                "label_en": "Proposed system (accurate)",
                "line_style": ":"
            },
            {
                "source": "mobilenet_025_same_per_layer_asymmetric_accurate",
                "color": "black",
                "marker": "x",
                "label": "stejná bitová přenost (accurate)",
                "label_en": "uniform bit precision (accurate)",
                "line_style": ":"
            }
        ],
    },
    {
        "title": "Mobilenet 0.25 (Per-layer asymetrická kvantizace)",
        "title_en": "Mobilenet 0.25 (Per-layer asymmetric quantization)",
        "output_file": "final_results_per_layer_asymmetric_025_accurate_difference",
        "base_memory": 1957472,
        "base_accuracy": 0.516,
        "data": [
            {
                "source": "mobilenet_025_qat_12_no_act_approx_per_layer_asymmetric_24pch_accurate_eval",
                "color": "tab:cyan",
                "marker": "s",
                "label": "Navržený systém (accurate)",
                "label_en": "Proposed system (accurate)",
                "line_style": ":",
                "include_before": False
            },
            {
                "source": "mobilenet_025_qat_12_no_act_approx_per_layer_asymmetric_24pch",
                "color": "tab:red",
                "marker": "o",
                "label": "Navržený systém",
                "label_en": "Proposed system",
                "line_style": ":",
                "include_before": False
            },
        ],
    },
    {
        "title": "Mobilenet 0.25 (Per-channel symetrická kvantizace)",
        "title_en": "MobileNet 0.25 (Per-channel symmetric quantization)",
        "output_file": "final_results_per_channel_symmetric_025_24h",
        "base_memory": 2047104,
        "base_accuracy": 0.516,
        "data": [
            {
                "source": "mobilenet_025_qat_12_no_act_per_channel_symmetric_24pch",
                "color": "tab:red",
                "marker": "o",
                "label": "Navržený systém",
                "label_en": "Proposed system",
                "line_style": ":"
            },
            {
                "source": "mobilenet_025_same_per_channel_symmetric",
                "color": "black",
                "marker": "x",
                "label": "stejná bitová přenost",
                "label_en": "uniform bit precision",
                "line_style": ":"
            },
        ],
    },
    {
        "title": "Mobilenet 0.25 (Per-channel symetrická kvantizace)",
        "title_en": "MobileNet 0.25 (Per-channel symmetric quantization)",
        "output_file": "final_results_per_channel_symmetric_025",
        "base_memory": 2047104,
        "base_accuracy": 0.516,
        "data": [
            {
                "source": "mobilenet_025_qat_12_no_act_per_channel_symmetric_24pch",
                "color": "tab:red",
                "marker": "o",
                "label": "Navržený systém (24h)",
                "label_en": "Proposed system",
                "line_style": ":"
            },
            {
                "source": "mobilenet_025_qat_12_no_act_per_channel_symmetric_24pch_48h",
                "color": "tab:cyan",
                "marker": "o",
                "label": "Navržený systém (48h)",
                "label_en": "Proposed system (48h)",
                "line_style": ":"
            },
            {
                "source": "mobilenet_025_same_per_channel_symmetric",
                "color": "black",
                "marker": "x",
                "label": "stejná bitová přenost",
                "label_en": "uniform bit precision",
                "line_style": ":"
            },
        ],
    }
]

for fig_def in figure_definitions:
    for i in range(2):
        en = i == 1
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))
        (ax1) = axes

        ax1.set_title(fig_def["title_en"] if en else fig_def["title"])

        if en:
            ax1.set_xlabel("Size of weights in comparison to 8bit model [%]")
            ax1.set_ylabel("Top-1 relative accuracy to original model [%]")
        else:
            ax1.set_xlabel("Velikost vah v porovnání s 8-bitovým modelem [%]")
            ax1.set_ylabel("Relativní Top-1 přesnost vůči původnímu modelu [%]")

        memory_8bit_ref = fig_def["base_memory"]
        accuracy_ref = fig_def["base_accuracy"]

        for data_def in fig_def["data"]:
            data = load_data("nsga_runs", data_def["source"], paretoFilter=True)
            # alldata += data
            label = data_def["label_en"] if en else data_def["label"]

            ax1.step([x["memory"] / memory_8bit_ref for x in data], [x["accuracy"] / accuracy_ref for x in data],
                     color=data_def["color"],
                     label=label, where="post", marker=data_def["marker"], markersize=5, linewidth=0.5,
                     linestyle=data_def["line_style"])

            if len(data) > 0 and "accuracy_before" in data[0] and (
                    "include_before" not in data_def or data_def["include_before"]):
                label_bef = (data_def["label_en"] + " (before fine-tuning)") if en else (
                        data_def["label"] + " (před dotrénováním)")
                ax1.step([x["memory"] / memory_8bit_ref for x in data],
                         [x["accuracy_before"] / accuracy_ref for x in data],
                         color=data_def["color"],
                         label=label_bef, where="post", marker=data_def["marker"],
                         markersize=5, linewidth=0.5,
                         linestyle=data_def["line_style"], alpha=0.25)

        ax1.set_ylim(0, 1.2)
        ax1.set_xlim(0, 1)

        ax1.xaxis.set_major_formatter(FuncFormatter(percent))
        ax1.yaxis.set_major_formatter(FuncFormatter(percent))

        ax1.legend()

        fig.savefig(f'fig/{fig_def["output_file"]}{"_en" if en else ""}.png', dpi=300)
        fig.savefig(f'fig/{fig_def["output_file"]}{"_en" if en else ""}.svg')

plt.show()
