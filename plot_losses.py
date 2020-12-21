import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,20)

def plot_metrics(metrics_treatment: List[Dict[str, List[float]]], metrics_baseline: List[Dict[str, List[float]]],
                 treatment_names: List[str], baseline_names: List[str], title: str):

    total_num_plots = len(metrics_treatment[0].keys()) * 100 + 10  # +100
    plt.suptitle(title, fontsize=32)
    for i, metric_name in enumerate(metrics_treatment[0].keys(), 1):
        plt.subplot(total_num_plots + i)
        plt.title(
            "Baseline: PODNet 1 example per class. Treatment: Podnet 1 example per class + SV entropy + Norm. (Entropy+norm)x100")
        plt.gca().set_title(metric_name)
        for i, name in enumerate(treatment_names):
            plt.plot(metrics_treatment[i][metric_name], label=name)
        for i, name in enumerate(baseline_names):
            plt.plot(metrics_baseline[i][metric_name], label=name)

        plt.legend()

    plt.legend()
    plt.show()

def read_json_to_dict(path: str) -> Dict[Any, Any]:
    with open(path, "r") as f:
        d = json.load(f)
    return d

def plot_single_metric(metric_array: List[float], title: str):

    plt.plot(metric_array, label=title, c="b")
    plt.legend()
    plt.show()

def plot_n_metric(metrics: List[List[float]], titles: List[str]):
    total_num_plots = len(metrics) * 100 + 10
    for i, (metric, title) in enumerate(zip(metrics, titles), 1):
        plt.subplot(total_num_plots+i)
        plt.title(title)
        plt.plot(metric, label='baseline',c="r")
    plt.show()


if __name__ == "__main__":
    baseline_paths = [
        "/home/leet/projects/La-MAML/logs/lamaml_cifar_baseline_w_acc_10memories/cifar100-2020-12-15_13-27-25-9703/0/losses.json",
                      "/home/leet/projects/La-MAML/logs/lamaml_cifar_baseline_w_acc_25memories/cifar100-2020-12-15_13-26-00-4497/0/losses.json",
#                      "/home/leet/projects/La-MAML/logs/lamaml_cifar_baseline_w_acc_50memories/cifar100-2020-12-13_22-07-29-2684/0/losses.json",
#                      "/home/leet/projects/La-MAML/logs/lamaml_cifar_baseline_w_acc_100memories/cifar100-2020-12-13_22-06-28-8768/0/losses.json"
                     ]
    treatment_paths = [
        "/home/leet/projects/La-MAML/logs/lamaml_cifar_sv_reg_with_ratio_norm_memories_10/cifar100-2020-12-15_13-26-27-0411/0/losses.json",
   #  "/home/leet/projects/La-MAML/logs/lamaml_cifar_sv_reg_with_ratio_norm_memories_25/cifar100-2020-12-15_13-25-36-3940/0/losses.json",
   #     "/home/leet/projects/La-MAML/logs/lamaml_cifar_sv_reg_with_ratio_norm_memories_50/cifar100-2020-12-13_22-03-44-9075/0/losses.json",
   #     "/home/leet/projects/La-MAML/logs/lamaml_cifar_sv_reg_with_ratio_norm_memories_100/cifar100-2020-12-13_22-03-50-4186/0/losses.json"
    ]
    FACTOR = 10
    treatment_losses = [read_json_to_dict(path) for path in treatment_paths]
    treatment_names = [f"sv + norm; memory: {factor}" for factor in [FACTOR, ]]
    baseline_losses = [read_json_to_dict(path) for path in baseline_paths]
    baseline_names = [f"baseline; memory: {factor}" for factor in [FACTOR, ]]
    TITLE = f"Memory {FACTOR} examples per class. Baseline vs SV(0.3) + Norm(0.05)\n Final Accuracy: " \
            f"Baseline: {0.6528}; SV + Norm {0.6616}"


    plot_metrics(treatment_losses, baseline_losses, treatment_names, baseline_names, title=TITLE)
    #plot_single_metric(treatment_losses['outer_ratio'], "treatment ratio")
    #plot_n_metric([baseline_losses['outer_ratio'], baseline_losses['accuracy']], ['outer_ratio', "accuracy"])