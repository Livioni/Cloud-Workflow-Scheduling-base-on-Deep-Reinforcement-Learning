"""Generate a Markdown report and SVG figures from sensitivity experiment CSVs."""

import argparse
import csv
import html
import statistics
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_policy_plot(rows, output):
    policies = ("fcfs", "sjf", "min_finish", "stale_min_finish")
    scenarios = sorted({int(row["scenario"]) for row in rows})
    values = {
        policy: [
            statistics.mean(
                float(row["makespan"]) for row in rows
                if int(row["scenario"]) == scenario and row["policy"] == policy
            )
            for scenario in scenarios
        ]
        for policy in policies
    }
    width, height = 1000, 560
    left, right, top, bottom = 75, 30, 55, 75
    all_values = [value for series in values.values() for value in series]
    low, high = min(all_values), max(all_values)
    if low == high:
        high = low + 1
    x_step = (width - left - right) / max(1, len(scenarios) - 1)
    colors = {
        "fcfs": "#2563eb",
        "sjf": "#16a34a",
        "min_finish": "#ea580c",
        "stale_min_finish": "#9333ea",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Mean makespan by sensitivity scenario</text>',
    ]
    for tick in range(6):
        y = top + tick * (height - top - bottom) / 5
        val = high - tick * (high - low) / 5
        elements.append(f'<path d="M {left} {y:.1f} H {width-right}" stroke="#ddd"/>')
        elements.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{val:.1f}</text>')
    for policy, series in values.items():
        points = []
        for index, value in enumerate(series):
            x = left + index * x_step
            y = top + (high - value) * (height - top - bottom) / (high - low)
            points.append(f"{x:.1f},{y:.1f}")
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[policy]}" stroke-width="2"/>'
        )
    for index, scenario in enumerate(scenarios):
        if index % max(1, len(scenarios) // 8) == 0:
            x = left + index * x_step
            elements.append(f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="10">{scenario}</text>')
    for index, policy in enumerate(policies):
        x = left + index * 205
        elements.append(f'<path d="M {x} {height-25} h 22" stroke="{colors[policy]}" stroke-width="3"/>')
        elements.append(f'<text x="{x+28}" y="{height-21}" font-family="sans-serif" font-size="12">{policy}</text>')
    elements.append("</svg>")
    Path(output).write_text("\n".join(elements), encoding="utf-8")


def write_heatmap(summary_rows, output):
    rows = [
        row for row in summary_rows
        if row["analysis"] == "policy_makespan"
        and row["policy"] == "min_finish"
    ]
    data_sizes = sorted({float(row["data_size_per_edge"]) for row in rows})
    link_scales = sorted({float(row["link_scale"]) for row in rows})
    delays = sorted({float(row["observation_delay"]) for row in rows})
    capacities = sorted({float(row["node_capacity_scale"]) for row in rows})
    matrix = {}
    for data_size in data_sizes:
        for link_scale in link_scales:
            values = [
                float(row["mean_makespan"]) for row in rows
                if float(row["data_size_per_edge"]) == data_size
                and float(row["link_scale"]) == link_scale
            ]
            matrix[data_size, link_scale] = statistics.mean(values)
    low, high = min(matrix.values()), max(matrix.values())
    width = max(540, 180 + len(link_scales) * 145)
    height = max(300, 150 + len(data_sizes) * 90)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="270" y="28" text-anchor="middle" font-family="sans-serif" font-size="16">Fresh min-finish makespan (averaged over delay/capacity levels)</text>',
    ]
    for col, link_scale in enumerate(link_scales):
        x = 155 + col * 145
        elements.append(f'<text x="{x+55}" y="68" text-anchor="middle" font-family="sans-serif" font-size="12">Link x{link_scale:g}</text>')
    for row_index, data_size in enumerate(data_sizes):
        y = 82 + row_index * 90
        elements.append(f'<text x="135" y="{y+38}" text-anchor="end" font-family="sans-serif" font-size="12">Data {data_size:g}</text>')
        for col, link_scale in enumerate(link_scales):
            value = matrix[data_size, link_scale]
            fraction = (value - low) / (high - low) if high > low else 0
            red = int(235 - 115 * fraction)
            green = int(248 - 100 * fraction)
            blue = int(255 - 40 * fraction)
            x = 155 + col * 145
            elements.append(f'<rect x="{x}" y="{y}" width="110" height="62" fill="rgb({red},{green},{blue})" stroke="white"/>')
            elements.append(f'<text x="{x+55}" y="{y+36}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:.1f}</text>')
    elements.append(f'<text x="20" y="{height-14}" font-family="sans-serif" font-size="10">Observation delays: {html.escape(", ".join(map(str, delays)))}; node capacity scales: {html.escape(", ".join(map(str, capacities)))}. Cell values average across those factors.</text>')
    elements.append("</svg>")
    Path(output).write_text("\n".join(elements), encoding="utf-8")


def generate_report(summary_csv, raw_csv, report_path, figure_dir):
    summary = read_csv(summary_csv)
    raw = read_csv(raw_csv)
    if not summary or not raw:
        raise ValueError("summary and raw CSV inputs must both contain records")
    policies = ("fcfs", "sjf", "min_finish", "stale_min_finish")
    scenario_count = len({row["scenario"] for row in raw})
    run_count = len(raw)
    all_tasks_completed = all(
        row["completed_tasks"] == row["task_count"] for row in raw
    )
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    policy_figure = figure_dir / "sensitivity_policy_makespan.svg"
    heatmap_figure = figure_dir / "sensitivity_network_heatmap.svg"
    write_policy_plot(raw, policy_figure)
    write_heatmap(summary, heatmap_figure)

    averages = {
        policy: statistics.mean(
            float(row["mean_makespan"]) for row in summary
            if row["analysis"] == "policy_makespan" and row["policy"] == policy
        )
        for policy in policies
    }
    stale_penalties = [
        float(row["mean_makespan"])
        for row in summary
        if row["analysis"] == "paired_makespan_difference"
        and row["policy"] == "stale_min_finish"
        and row["comparison_policy"] == "min_finish"
    ]
    penalty_mean = statistics.mean(stale_penalties)
    data_sizes = sorted({float(row["data_size_per_edge"]) for row in raw})
    delays = sorted({float(row["observation_delay"]) for row in raw})
    links = sorted({float(row["link_scale"]) for row in raw})
    capacities = sorted({float(row["node_capacity_scale"]) for row in raw})
    factor_sections = []
    factors = (
        ("data_size_per_edge", "Payload size"),
        ("observation_delay", "Observation delay"),
        ("link_scale", "Link scale"),
        ("node_capacity_scale", "Node capacity scale"),
    )
    for factor, title in factors:
        factor_sections.append(f"### {title}\n\n| Level | " + " | ".join(policies) + " |")
        factor_sections.append("|---:|" + "---:|" * len(policies))
        levels = sorted({float(row[factor]) for row in raw})
        for level in levels:
            means = [
                statistics.mean(
                    float(row["makespan"]) for row in raw
                    if float(row[factor]) == level and row["policy"] == policy
                )
                for policy in policies
            ]
            factor_sections.append(
                f"| {level:g} | " + " | ".join(f"{value:.3f}" for value in means) + " |"
            )
    report = f"""# Edge–Cloud Scheduling Sensitivity Analysis

## Executive summary

This report documents a controlled simulator sensitivity run, not a live Kubernetes experiment. It evaluates **{scenario_count}** configurations and **{run_count:,}** policy/workflow runs across the repository's 1,000 50-task test-array entries. All tasks completed in every recorded run: **{all_tasks_completed}**.

Across scenarios, the unweighted average of scenario mean makespans was FCFS **{averages['fcfs']:.3f}**, SJF **{averages['sjf']:.3f}**, fresh minimum-finish **{averages['min_finish']:.3f}**, and stale minimum-finish **{averages['stale_min_finish']:.3f}**. The mean paired stale-vs-fresh minimum-finish penalty across scenarios was **{penalty_mean:.3f}** time units. Scenario-averaging is descriptive; it does not pool individual DAGs into one inference.

## Research question and design

The experiment asks how heuristic makespan changes under selected assumptions about dependency payload size, network conditions, node capacity, and stale monitoring. Every policy sees the same workflow and resource configuration within a scenario. Fresh FCFS, SJF, and minimum-finish use exact current capacity; stale minimum-finish uses the configured delayed snapshot. Fixed seed base was 2300; there was one deterministic duration replicate per DAG, and no duration jitter.

The base cluster configuration in `configs/edge_cloud_dataset.json` has two nodes, each with CPU=100 and memory=100; Edge speed=0.8 and Cloud speed=1.5. The base link has latency=5 and bandwidth=10 in simulator units. Sensitivity scaling modifies these abstract inputs; none are measurements from physical nodes or links.

The factorial axes were:

| Factor | Tested values | Meaning |
|---|---:|---|
| Uniform data size per dependency | {', '.join(map(str, data_sizes))} | Synthetic payload units assigned to each DAG edge |
| Observation refresh delay | {', '.join(map(str, delays))} | Simulator time units before stale free-capacity snapshot refresh |
| Link scale | {', '.join(map(str, links))} | Multiplies latency and divides bandwidth; higher means slower links |
| Node capacity scale | {', '.join(map(str, capacities))} | Multiplies each node's CPU and memory capacities |

This is a {scenario_count}-cell full factorial grid. Each cell ran the same 1,000 DAG indices and all four policies. Bootstrap summaries use 500 resamples for descriptive intervals; the primary sensitivity comparison is the matched per-DAG output, available in the raw CSV.

## Results

| Policy | Mean of scenario means (time units) |
|---|---:|
| FCFS | {averages['fcfs']:.3f} |
| SJF | {averages['sjf']:.3f} |
| Fresh minimum-finish | {averages['min_finish']:.3f} |
| Stale minimum-finish | {averages['stale_min_finish']:.3f} |

The following marginal summaries average over all other grid dimensions. They are descriptive summaries, not a causal estimate outside these selected values.

{chr(10).join(factor_sections)}

The stale-vs-fresh comparison varies across the grid. The network heatmap averages fresh minimum-finish makespan over observation-delay and capacity levels for each data-size/link-scale pair; the policy figure plots the scenario means.

![Policy makespan across sensitivity scenarios](results/{policy_figure.name})

![Network and payload sensitivity heatmap](results/{heatmap_figure.name})

## Artifacts and reproduction

- Summary CSV: `{Path(summary_csv).as_posix()}`
- Per-workflow raw CSV: `{Path(raw_csv).as_posix()}`
- Run log: `results/sensitivity_run.log`
- Figures: `{policy_figure.as_posix()}` and `{heatmap_figure.as_posix()}`
- Configuration: `configs/edge_cloud_dataset.json`
- Test suite: `python -m unittest discover -s tests -v`

Re-run the grid from the repository root in PowerShell:

```powershell
$python = Join-Path $env:LOCALAPPDATA 'venvs\\cwscheduling-py312\\Scripts\\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$indices = 0..999
& $python edge_cloud_experiment.py --config configs\\edge_cloud_dataset.json --dataset-size 50 --dataset-split test --dataset-indices $indices --replicates 1 --seed 2300 --bootstrap-samples 500 --output results\\sensitivity_test50.csv --sensitivity-raw-output results\\sensitivity_test50_raw.csv --log-output results\\sensitivity_run.log --sensitivity-data-sizes 1 10 --sensitivity-observation-delays 0 5 --sensitivity-link-scales 0.5 2 --sensitivity-node-capacity-scales 0.8 1.2
& $python generate_sensitivity_report.py --summary results\\sensitivity_test50.csv --raw results\\sensitivity_test50_raw.csv --report SENSITIVITY_ANALYSIS_REPORT.md --figure-dir results
```

## Limitations and interpretation

The repository does not establish that `npy/test_datasheet/50/` is an independently sampled or officially held-out split; its provenance remains undocumented. DAG arrays contain no edge payload volumes, so payload sizes are synthetic and uniform. Node capacities, link latency/bandwidth, scaling levels, and observation delay are simulation assumptions, not hardware/network measurements. CPU and memory are abstract units. The transfer model has no bandwidth contention or overlap, and stale observations omit noise, loss, and confidence. The policy set is small and the objective is makespan only.

Therefore, the experiment demonstrates sensitivity of this implementation to explicit parameters. It does **not** establish real-cluster behavior, a statistically general policy ranking, or a thesis contribution by itself. The unweighted averages across different parameter cells are summaries of the chosen grid, not a natural deployment distribution. Select parameter ranges from literature, measurement, or an explicitly justified scenario model before drawing research conclusions.

## Implementation and validation status

The simulator currently supports heterogeneous abstract nodes, DAG precedence, resource reservations, task-scoped pod records, configurable link delays, delayed free-capacity snapshots, and four heuristic variants. The generated artifacts preserve both scenario-level summaries and individual DAG/policy rows. The model remains a research prototype; reusable Kubernetes pods, multi-application arrivals, live Kubernetes, noisy telemetry, and a simulator-specific RL policy are not implemented.
"""
    Path(report_path).write_text(report, encoding="utf-8")
    return policy_figure, heatmap_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/sensitivity_test50.csv")
    parser.add_argument("--raw", default="results/sensitivity_test50_raw.csv")
    parser.add_argument("--report", default="SENSITIVITY_ANALYSIS_REPORT.md")
    parser.add_argument("--figure-dir", default="results")
    args = parser.parse_args()
    figures = generate_report(args.summary, args.raw, args.report, args.figure_dir)
    print(f"report written to {args.report}")
    for figure in figures:
        print(f"figure written to {figure}")


if __name__ == "__main__":
    main()
