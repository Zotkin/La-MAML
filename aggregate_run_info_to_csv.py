import os
from typing import List, Dict, Any
import json
import re

import pandas as pd


def get_final_accuracy_from_results_txt(lines: List[str]):
    for line in lines:
        line = line.strip().split(" ")
        if "Final" in line:
            return float(line[-1])
    raise ValueError("No line beginning fith 'Final' found in document.")


def collect_logs_from_dir(path: str) -> List[Dict[str, Any]]:
    experiments_paths = [os.path.join(path, filename) for filename in os.listdir(path)]
    logs = []
    for path in experiments_paths:
        if os.path.isdir(path):
            logs.extend(collect_logs_recursively(path))

    return logs


def collect_logs_recursively(path: str) -> List[Dict[str, Any]]:
    results = []
    filenames = os.listdir(path)
    if "results.txt" in filenames:
        r = parse_results_folder(path)
        results.append(r)
    elif "tfdir" in filenames:
        # folder contains tfdir but not results indicate unfinished experiment. Skip
        return []
    else:
        for fname in filenames:
            r = collect_logs_recursively(os.path.join(path, fname))
            if r is not None:
                results.extend(r)
    return results


def parse_results_folder(path: str) -> Dict[str, Any]:

    with open(os.path.join(path, "results.txt"), "r") as f:
        lines = f.readlines()
        final_accuracy = get_final_accuracy_from_results_txt(lines)

    with open(os.path.join(path, "training_parameters.json"), "r") as f:
        results_dict = json.load(f)

    results_dict["final_accuracy"] = final_accuracy
    results_dict['log_path'] = path
    results_dict['timestamp'] = parse_timestamp_from_path(path)
    return results_dict


def parse_timestamp_from_path(path: str) -> str:
    pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{4}'
    match = re.findall(pattern, path)
    if len(match) != 1:
        raise Warning(f"When attempting to find match timestamp in path, {len(match)} we  found, expected 1.")
    return match[0]




if __name__ == "__main__":
    path = "/home/leet/projects/cvpr21/logs/lamaml"
    logs = collect_logs_from_dir(path)
    logs_table = pd.DataFrame(logs)
    logs_table.to_csv(os.path.join(path,"la-maml-aggrerated-results-27-12-2020.csv"))