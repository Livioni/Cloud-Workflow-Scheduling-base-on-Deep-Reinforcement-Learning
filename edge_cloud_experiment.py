"""Run deterministic FCFS/SJF experiments from a JSON workload configuration."""

import argparse
import csv
import json
import logging
import math
import random
import statistics
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from Envs.edge_cloud_simulator import (
    Dependency,
    EdgeCloudSimulator,
    Link,
    Node,
    Task,
)


def load_config(path):
    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)
    nodes = [Node(**item) for item in config["nodes"]]
    tasks = [
        Task(
            **{
                **item,
                "allowed_tiers": tuple(item.get("allowed_tiers", ())),
            }
        )
        for item in config["tasks"]
    ]
    dependencies = [Dependency(**item) for item in config.get("dependencies", [])]
    links = {
        tuple(item["nodes"]): Link(item["latency"], item["bandwidth"])
        for item in config.get("links", [])
    }
    return nodes, tasks, dependencies, links


def load_repository_workload(
    size, index, data_size_per_edge=0.0, repository_root=".", split="train"
):
    """Load one committed DAG sample; edge payload size is an explicit assumption."""
    if data_size_per_edge < 0:
        raise ValueError("data_size_per_edge must be non-negative")
    root = Path(repository_root)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if split == "train":
        dataset = root / "npy" / "train_datasheet" / str(size)
        suffix = str(size)
    else:
        dataset = root / "npy" / "test_datasheet" / str(size)
        suffix = str(size)
    edges = np.load(dataset / f"edges{suffix}_lib.npy", allow_pickle=True).tolist()
    durations = np.load(dataset / f"duration{suffix}_lib.npy", allow_pickle=True)
    demands = np.load(dataset / f"demand{suffix}_lib.npy", allow_pickle=True)
    if index < 0 or index >= len(durations):
        raise IndexError(f"workload index {index} outside [0, {len(durations) - 1}]")
    tasks = [
        Task(str(i + 1), float(durations[index][i]),
             float(demands[index][i][0]), float(demands[index][i][1]))
        for i in range(len(durations[index]))
    ]
    dependencies = [
        Dependency(str(parent), str(child), data_size_per_edge)
        for parent, child in edges[index]
        if parent != "Start" and child != "Exit"
    ]
    return tasks, dependencies


def run_experiments(
    config_path, output_path, replicates=1, seed=0, duration_jitter=0.0,
    summary_path=None, bootstrap_samples=2000, dataset_size=None,
    dataset_index=0, data_size_per_edge=0.0, dataset_indices=None,
    dataset_split="train",
    observation_delay=0.0,
):
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    if duration_jitter < 0 or duration_jitter >= 1:
        raise ValueError("duration_jitter must be in [0, 1)")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least one")
    nodes, tasks, dependencies, links = load_config(config_path)
    if dataset_indices is None:
        dataset_indices = [dataset_index]
    if dataset_size is None and len(dataset_indices) != 1:
        raise ValueError("dataset-size is required when multiple dataset indices are requested")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_sample in dataset_indices:
        if dataset_size is None:
            sample_tasks, sample_dependencies = tasks, dependencies
        else:
            sample_tasks, sample_dependencies = load_repository_workload(
                dataset_size, dataset_sample, data_size_per_edge, split=dataset_split
            )
        for replicate in range(replicates):
            run_seed = seed + replicate
            rng = random.Random(run_seed)
            run_tasks = [
                replace(
                    task,
                    duration=task.duration * rng.uniform(
                        1 - duration_jitter, 1 + duration_jitter
                    ),
                )
                for task in sample_tasks
            ]
            for policy in ("fcfs", "sjf", "min_finish", "stale_min_finish"):
                simulator = EdgeCloudSimulator(
                    nodes, run_tasks, sample_dependencies, links,
                    observation_delay=(
                        observation_delay if policy == "stale_min_finish" else 0.0
                    ),
                )
                metrics = simulator.run(policy)
                if metrics["completed_tasks"] != metrics["task_count"]:
                    raise RuntimeError(f"{policy} did not complete all tasks")
                if simulator.completed != set(simulator.tasks):
                    raise RuntimeError(f"{policy} completion set is inconsistent")
                if any(
                    not math.isclose(state.available_cpu, state.spec.cpu_capacity, rel_tol=1e-12)
                    or not math.isclose(
                        state.available_memory, state.spec.memory_capacity, rel_tol=1e-12
                    )
                    for state in simulator.nodes.values()
                ):
                    raise RuntimeError(f"{policy} did not release all node resources")
                rows.append({
                    "dataset_index": dataset_sample,
                    "replicate": replicate,
                    "seed": run_seed,
                    "policy": policy,
                    "makespan": metrics["makespan"],
                    "completed_tasks": metrics["completed_tasks"],
                    "task_count": metrics["task_count"],
                    "communication_delay": metrics["communication_delay"],
                    "transferred_bytes": metrics["transferred_bytes"],
                    "stale_observation_decisions": metrics["stale_observation_decisions"],
                    "observation_wait_time": metrics["observation_wait_time"],
                    "cpu_utilization": json.dumps(
                        {key: value["cpu"] for key, value in metrics["node_utilization"].items()},
                        sort_keys=True,
                    ),
                    "memory_utilization": json.dumps(
                        {key: value["memory"] for key, value in metrics["node_utilization"].items()},
                        sort_keys=True,
                    ),
                })
    with output.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = Path(summary_path) if summary_path else output.with_name(
        f"{output.stem}_summary{output.suffix}"
    )
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "analysis", "policy", "comparison_policy", "replicates",
        "mean_makespan", "sample_stdev_makespan",
        "ci95_low", "ci95_high", "ci_method",
    ]
    with summary.open("w", newline="", encoding="utf-8") as summary_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        summary_writer.writeheader()
        policies = ("fcfs", "sjf", "min_finish", "stale_min_finish")
        for policy in policies:
            values = [row["makespan"] for row in rows if row["policy"] == policy]
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            if len(values) > 1:
                bootstrap_rng = random.Random(seed + sum(map(ord, policy)))
                means = sorted(
                    statistics.mean(
                        bootstrap_rng.choices(values, k=len(values))
                    )
                    for _ in range(bootstrap_samples)
                )
                low_index = int(0.025 * bootstrap_samples)
                high_index = min(
                    bootstrap_samples - 1, int(0.975 * bootstrap_samples)
                )
                ci_low, ci_high = means[low_index], means[high_index]
            else:
                ci_low = ci_high = mean
            summary_writer.writerow({
                "analysis": "policy_makespan",
                "policy": policy,
                "replicates": len(values),
                "mean_makespan": mean,
                "sample_stdev_makespan": stdev,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "ci_method": "percentile bootstrap over replicate makespans",
            })
        reference = {
            (row["dataset_index"], row["replicate"]): row["makespan"]
            for row in rows if row["policy"] == "fcfs"
        }
        for policy in policies[1:]:
            paired_differences = [
                row["makespan"] - reference[(row["dataset_index"], row["replicate"])]
                for row in rows if row["policy"] == policy
            ]
            mean_difference = statistics.mean(paired_differences)
            stdev_difference = (
                statistics.stdev(paired_differences)
                if len(paired_differences) > 1 else 0.0
            )
            if len(paired_differences) > 1:
                bootstrap_rng = random.Random(seed + sum(map(ord, policy)) + 1)
                bootstrap_means = sorted(
                    statistics.mean(
                        bootstrap_rng.choices(paired_differences, k=len(paired_differences))
                    )
                    for _ in range(bootstrap_samples)
                )
                low_index = int(0.025 * bootstrap_samples)
                high_index = min(
                    bootstrap_samples - 1, int(0.975 * bootstrap_samples)
                )
                ci_low, ci_high = bootstrap_means[low_index], bootstrap_means[high_index]
            else:
                ci_low = ci_high = mean_difference
            summary_writer.writerow({
                "analysis": "paired_makespan_difference",
                "policy": policy,
                "comparison_policy": "fcfs",
                "replicates": len(paired_differences),
                "mean_makespan": mean_difference,
                "sample_stdev_makespan": stdev_difference,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "ci_method": "paired percentile bootstrap; difference vs FCFS",
            })
        fresh_min_finish = {
            (row["dataset_index"], row["replicate"]): row["makespan"]
            for row in rows if row["policy"] == "min_finish"
        }
        stale_differences = [
            row["makespan"] - fresh_min_finish[(row["dataset_index"], row["replicate"])]
            for row in rows if row["policy"] == "stale_min_finish"
        ]
        stale_mean = statistics.mean(stale_differences)
        if len(stale_differences) > 1:
            stale_rng = random.Random(seed + sum(map(ord, "stale_min_finish")) + 2)
            stale_bootstrap = sorted(
                statistics.mean(stale_rng.choices(stale_differences, k=len(stale_differences)))
                for _ in range(bootstrap_samples)
            )
            stale_low = stale_bootstrap[int(0.025 * bootstrap_samples)]
            stale_high = stale_bootstrap[min(
                bootstrap_samples - 1, int(0.975 * bootstrap_samples)
            )]
            stale_stdev = statistics.stdev(stale_differences)
        else:
            stale_low = stale_high = stale_mean
            stale_stdev = 0.0
        summary_writer.writerow({
            "analysis": "paired_makespan_difference",
            "policy": "stale_min_finish",
            "comparison_policy": "min_finish",
            "replicates": len(stale_differences),
            "mean_makespan": stale_mean,
            "sample_stdev_makespan": stale_stdev,
            "ci95_low": stale_low,
            "ci95_high": stale_high,
            "ci_method": "paired percentile bootstrap; stale observation vs fresh observation",
        })
    return rows, summary


def run_sensitivity(
    config_path,
    output_path,
    data_sizes,
    observation_delays,
    link_scales,
    node_capacity_scales,
    raw_output_path=None,
    log_path=None,
    **experiment_options,
):
    """Run a Cartesian sensitivity grid and combine per-scenario summaries."""
    axes = {
        "data_size_per_edge": data_sizes,
        "observation_delay": observation_delays,
        "link_scale": link_scales,
        "node_capacity_scale": node_capacity_scales,
    }
    if any(not values for values in axes.values()):
        raise ValueError("sensitivity axes must not be empty")
    if any(value < 0 for value in data_sizes):
        raise ValueError("data sizes must be non-negative")
    if any(value < 0 for value in observation_delays):
        raise ValueError("observation delays must be non-negative")
    if any(not math.isfinite(value) or value <= 0 for value in link_scales):
        raise ValueError("link scales must be finite and positive")
    if any(not math.isfinite(value) or value <= 0 for value in node_capacity_scales):
        raise ValueError("node capacity scales must be finite and positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "data_size_per_edge", "observation_delay", "link_scale",
        "node_capacity_scale", "analysis", "policy", "comparison_policy",
        "replicates", "mean_makespan", "sample_stdev_makespan",
        "ci95_low", "ci95_high", "ci_method",
    ]
    results = []
    raw_rows = []
    logger = logging.getLogger("edge_cloud_sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    else:
        handler = None
    logger.info(
        "experiment start config=%s data_sizes=%s observation_delays=%s "
        "link_scales=%s node_capacity_scales=%s options=%s",
        config_path, data_sizes, observation_delays, link_scales,
        node_capacity_scales, experiment_options,
    )
    with tempfile.TemporaryDirectory() as temporary:
        temp_root = Path(temporary)
        with Path(config_path).open(encoding="utf-8") as source:
            base_config = json.load(source)
        scenario = 0
        for data_size in data_sizes:
            for delay in observation_delays:
                for link_scale in link_scales:
                    for capacity_scale in node_capacity_scales:
                        scenario += 1
                        config = json.loads(json.dumps(base_config))
                        for node in config["nodes"]:
                            node["cpu_capacity"] *= capacity_scale
                            node["memory_capacity"] *= capacity_scale
                        for link in config.get("links", []):
                            link["latency"] *= link_scale
                            link["bandwidth"] /= link_scale
                        scenario_config = temp_root / f"config-{scenario}.json"
                        scenario_config.write_text(
                            json.dumps(config), encoding="utf-8"
                        )
                        raw_path = temp_root / f"raw-{scenario}.csv"
                        logger.info(
                            "scenario=%s data_size=%s observation_delay=%s "
                            "link_scale=%s node_capacity_scale=%s start",
                            scenario, data_size, delay, link_scale, capacity_scale,
                        )
                        scenario_rows, summary_path = run_experiments(
                            scenario_config,
                            raw_path,
                            data_size_per_edge=data_size,
                            observation_delay=delay,
                            **experiment_options,
                        )
                        for row in scenario_rows:
                            raw_rows.append({
                                "scenario": scenario,
                                "data_size_per_edge": data_size,
                                "observation_delay": delay,
                                "link_scale": link_scale,
                                "node_capacity_scale": capacity_scale,
                                **row,
                            })
                        with summary_path.open(newline="", encoding="utf-8") as summary_file:
                            for row in csv.DictReader(summary_file):
                                results.append({
                                    "data_size_per_edge": data_size,
                                    "observation_delay": delay,
                                    "link_scale": link_scale,
                                    "node_capacity_scale": capacity_scale,
                                    **row,
                                })
                        logger.info("scenario=%s completed runs=%s", scenario, len(scenario_rows))
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    if raw_output_path:
        raw_output = Path(raw_output_path)
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        if raw_rows:
            with raw_output.open("w", newline="", encoding="utf-8") as raw_file:
                writer = csv.DictWriter(raw_file, fieldnames=list(raw_rows[0]))
                writer.writeheader()
                writer.writerows(raw_rows)
    if handler:
        logger.info(
            "experiment complete scenarios=%s summary_rows=%s raw_rows=%s",
            scenario, len(results), len(raw_rows),
        )
        logger.removeHandler(handler)
        handler.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/edge_cloud_example.json")
    parser.add_argument("--output", default="results/edge_cloud_metrics.csv")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-jitter", type=float, default=0.0)
    parser.add_argument("--summary-output")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--dataset-indices", nargs="+", type=int)
    parser.add_argument("--dataset-split", choices=("train", "test"), default="train")
    parser.add_argument("--data-size-per-edge", type=float, default=0.0)
    parser.add_argument("--observation-delay", type=float, default=0.0)
    parser.add_argument("--sensitivity-data-sizes", nargs="+", type=float)
    parser.add_argument("--sensitivity-observation-delays", nargs="+", type=float)
    parser.add_argument("--sensitivity-link-scales", nargs="+", type=float)
    parser.add_argument("--sensitivity-node-capacity-scales", nargs="+", type=float)
    parser.add_argument("--sensitivity-raw-output")
    parser.add_argument("--log-output")
    args = parser.parse_args()
    sensitivity_axes = (
        args.sensitivity_data_sizes,
        args.sensitivity_observation_delays,
        args.sensitivity_link_scales,
        args.sensitivity_node_capacity_scales,
    )
    if any(axis is not None for axis in sensitivity_axes):
        if not all(axis is not None for axis in sensitivity_axes):
            parser.error("all four --sensitivity-* axes must be specified together")
        results = run_sensitivity(
            args.config,
            args.output,
            args.sensitivity_data_sizes,
            args.sensitivity_observation_delays,
            args.sensitivity_link_scales,
            args.sensitivity_node_capacity_scales,
            raw_output_path=args.sensitivity_raw_output,
            log_path=args.log_output,
            replicates=args.replicates,
            seed=args.seed,
            duration_jitter=args.duration_jitter,
            bootstrap_samples=args.bootstrap_samples,
            dataset_size=args.dataset_size,
            dataset_index=args.dataset_index,
            dataset_indices=args.dataset_indices,
            dataset_split=args.dataset_split,
        )
        print(f"sensitivity rows={len(results)} written to {args.output}")
        return
    rows, summary = run_experiments(
        args.config, args.output, args.replicates, args.seed, args.duration_jitter,
        args.summary_output, args.bootstrap_samples, args.dataset_size,
        args.dataset_index, args.data_size_per_edge, args.dataset_indices,
        args.dataset_split,
        args.observation_delay,
    )
    for row in rows:
        print(
            f"replicate={row['replicate']} seed={row['seed']} {row['policy']}: "
            f"makespan={row['makespan']}, tasks={row['completed_tasks']}/{row['task_count']}"
        )
    print(f"metrics written to {args.output}")
    print(f"summary written to {summary}")


if __name__ == "__main__":
    main()
