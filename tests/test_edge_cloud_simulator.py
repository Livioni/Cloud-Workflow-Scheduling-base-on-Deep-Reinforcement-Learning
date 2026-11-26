import unittest
import csv
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from edge_cloud_experiment import (
    load_repository_workload,
    run_experiments,
    run_sensitivity,
)
from generate_sensitivity_report import generate_report
from Envs.edge_cloud_simulator import (
    Dependency,
    EdgeCloudSimulator,
    Link,
    Node,
    Task,
)


class EdgeCloudSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            Node("edge-1", "edge", 4, 8, speed=1),
            Node("cloud-1", "cloud", 8, 16, speed=2),
        ]

    def test_schedules_dag_and_releases_resources(self):
        simulator = EdgeCloudSimulator(
            self.nodes,
            [Task("a", 8, 2, 4), Task("b", 4, 2, 4)],
            [Dependency("a", "b")],
        )
        metrics = simulator.run()
        self.assertEqual(metrics["completed_tasks"], 2)
        self.assertEqual(metrics["makespan"], 6)
        self.assertEqual(metrics["task_start_times"]["b"], 4)
        self.assertEqual(simulator.nodes["edge-1"].available_cpu, 4)
        self.assertEqual(simulator.nodes["edge-1"].available_memory, 8)
        self.assertEqual(
            {pod.status for pod in simulator.pods.values()}, {"completed"}
        )
        self.assertEqual(
            {pod.node_id for pod in simulator.pods.values()},
            set(metrics["placements"].values()),
        )
        self.assertLessEqual(metrics["node_utilization"]["edge-1"]["cpu"], 1)
        self.assertLessEqual(metrics["node_utilization"]["edge-1"]["memory"], 1)

    def test_network_delay_is_applied_to_precedence(self):
        simulator = EdgeCloudSimulator(
            [Node("edge-1", "edge", 4, 8), Node("cloud-1", "cloud", 2, 8)],
            [Task("a", 2, 4, 2), Task("b", 2, 2, 2)],
            [Dependency("a", "b", data_size=8)],
            {("edge-1", "cloud-1"): Link(latency=1, bandwidth=4)},
        )
        metrics = simulator.run()
        self.assertEqual(metrics["task_finish_times"]["a"], 2)
        self.assertEqual(metrics["task_start_times"]["b"], 5)
        self.assertEqual(metrics["makespan"], 7)
        self.assertEqual(metrics["communication_delay"], 3)
        self.assertEqual(metrics["transferred_bytes"], 8)

    def test_multi_parent_child_waits_for_slowest_parent_transfer(self):
        nodes = [
            Node("edge-a", "edge", 2, 4),
            Node("edge-b", "edge", 2, 4),
            Node("cloud", "cloud", 2, 4),
        ]
        tasks = [
            Task("parent-a", 2, 2, 1, allowed_tiers=("edge",)),
            Task("parent-b", 4, 2, 1, allowed_tiers=("edge",)),
            Task("child", 1, 2, 1, allowed_tiers=("cloud",)),
        ]
        dependencies = [
            Dependency("parent-a", "child", 4),
            Dependency("parent-b", "child", 8),
        ]
        links = {
            ("edge-a", "cloud"): Link(1, 4),
            ("edge-b", "cloud"): Link(2, 4),
        }
        metrics = EdgeCloudSimulator(nodes, tasks, dependencies, links).run()
        self.assertEqual(metrics["task_finish_times"]["parent-a"], 2)
        self.assertEqual(metrics["task_finish_times"]["parent-b"], 4)
        self.assertEqual(metrics["task_start_times"]["child"], 8)
        self.assertEqual(metrics["makespan"], 9)

    def test_transfer_release_can_precede_an_unrelated_task_completion(self):
        nodes = [
            Node("edge", "edge", 4, 4),
            Node("cloud", "cloud", 2, 2),
        ]
        tasks = [
            Task("parent", 2, 1, 1, allowed_tiers=("edge",)),
            Task("unrelated", 10, 1, 1, allowed_tiers=("edge",)),
            Task("child", 1, 1, 1, allowed_tiers=("cloud",)),
        ]
        deps = [Dependency("parent", "child", data_size=1)]
        links = {("edge", "cloud"): Link(latency=1, bandwidth=1)}
        metrics = EdgeCloudSimulator(nodes, tasks, deps, links).run()
        self.assertEqual(metrics["task_start_times"]["child"], 4)
        self.assertEqual(metrics["task_finish_times"]["unrelated"], 10)
        self.assertEqual(metrics["makespan"], 10)

    def test_sjf_changes_order_for_ready_tasks(self):
        nodes = [Node("edge", "edge", 2, 2)]
        tasks = [Task("a-long", 5, 2, 2), Task("z-short", 1, 2, 2)]
        fcfs = EdgeCloudSimulator(nodes, tasks, []).run("fcfs")
        sjf = EdgeCloudSimulator(nodes, tasks, []).run("sjf")
        self.assertEqual(fcfs["task_finish_times"]["a-long"], 5)
        self.assertEqual(sjf["task_finish_times"]["z-short"], 1)

    def test_tier_constraints_are_respected_or_report_unschedulable(self):
        tasks = [Task("edge-only", 2, 1, 1, allowed_tiers=("edge",))]
        metrics = EdgeCloudSimulator(self.nodes, tasks, []).run()
        self.assertEqual(metrics["placements"]["edge-only"], "edge-1")

        impossible = [Task("cloud-only", 2, 1, 1, allowed_tiers=("cloud",))]
        nodes = [Node("edge-only-node", "edge", 4, 8)]
        with self.assertRaisesRegex(RuntimeError, "unschedulable"):
            EdgeCloudSimulator(nodes, impossible, []).run()

    def test_zero_duration_tasks_are_supported(self):
        simulator = EdgeCloudSimulator(
            self.nodes,
            [Task("instant", 0, 1, 1), Task("next", 2, 1, 1)],
            [Dependency("instant", "next")],
        )
        metrics = simulator.run()
        self.assertEqual(metrics["task_finish_times"]["instant"], 0)
        self.assertEqual(metrics["task_start_times"]["next"], 0)
        self.assertEqual(metrics["makespan"], 1)

    def test_delayed_observation_waits_before_reusing_released_capacity(self):
        simulator = EdgeCloudSimulator(
            [Node("edge", "edge", 2, 2)],
            [Task("first", 2, 2, 2), Task("second", 1, 2, 2)],
            [],
            observation_delay=5,
        )
        metrics = simulator.run("stale_min_finish")
        self.assertEqual(metrics["task_start_times"]["first"], 5)
        self.assertEqual(metrics["makespan"], 7)
        self.assertEqual(metrics["observation_wait_time"], 4)
        self.assertEqual(metrics["stale_observation_decisions"], 1)

    def test_independent_tasks_run_concurrently_with_capacity_accounting(self):
        nodes = [Node("edge", "edge", 4, 8)]
        tasks = [Task("first", 4, 2, 4), Task("second", 4, 2, 4)]
        simulator = EdgeCloudSimulator(nodes, tasks, [])
        metrics = simulator.run()
        self.assertEqual(metrics["makespan"], 4)
        self.assertEqual(metrics["task_start_times"]["first"], 0)
        self.assertEqual(metrics["task_start_times"]["second"], 0)
        self.assertEqual(metrics["node_utilization"]["edge"]["cpu"], 1)
        self.assertEqual(metrics["node_utilization"]["edge"]["memory"], 1)

    def test_simulator_cannot_be_run_twice(self):
        simulator = EdgeCloudSimulator(
            self.nodes, [Task("once", 1, 1, 1)], []
        )
        simulator.run()
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            simulator.run()

    def test_rejects_cycles_and_missing_links(self):
        with self.assertRaisesRegex(ValueError, "DAG"):
            EdgeCloudSimulator(
                self.nodes,
                [Task("a", 1, 1, 1), Task("b", 1, 1, 1)],
                [Dependency("a", "b"), Dependency("b", "a")],
            )
        simulator = EdgeCloudSimulator(
            [Node("edge-1", "edge", 4, 8), Node("cloud-1", "cloud", 2, 8)],
            [Task("a", 1, 4, 1), Task("b", 1, 2, 1)],
            [Dependency("a", "b", data_size=1)],
        )
        with self.assertRaisesRegex(ValueError, "missing network link"):
            simulator.run()

    def test_rejects_task_that_cannot_fit_any_node(self):
        simulator = EdgeCloudSimulator(
            self.nodes, [Task("too-large", 1, 100, 1)], []
        )
        with self.assertRaisesRegex(RuntimeError, "unschedulable"):
            simulator.run()

    def test_experiment_runner_writes_both_policy_rows(self):
        config = Path("configs/edge_cloud_example.json")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "metrics.csv"
            rows, summary = run_experiments(config, output)
            self.assertEqual(
                [row["policy"] for row in rows],
                ["fcfs", "sjf", "min_finish", "stale_min_finish"],
            )
            self.assertEqual(float(rows[0]["communication_delay"]), 17)
            self.assertEqual(float(rows[0]["transferred_bytes"]), 14)
            with output.open(newline="", encoding="utf-8") as result_file:
                written = list(csv.DictReader(result_file))
            self.assertEqual(len(written), 4)
            self.assertEqual(written[0]["completed_tasks"], written[0]["task_count"])
            self.assertTrue(summary.exists())
            with summary.open(newline="", encoding="utf-8") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
            self.assertEqual(len(summary_rows), 8)
            self.assertTrue(all(row["ci_method"] for row in summary_rows))
            self.assertEqual(
                [row["analysis"] for row in summary_rows[-2:]],
                ["paired_makespan_difference", "paired_makespan_difference"],
            )

    def test_seeded_replicates_are_reproducible_and_paired(self):
        config = Path("configs/edge_cloud_example.json")
        with tempfile.TemporaryDirectory() as temporary:
            first, _ = run_experiments(
                config, Path(temporary) / "first.csv", replicates=3,
                seed=17, duration_jitter=0.1,
            )
            second, _ = run_experiments(
                config, Path(temporary) / "second.csv", replicates=3,
                seed=17, duration_jitter=0.1,
            )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        for replicate in range(3):
            rows = [row for row in first if row["replicate"] == replicate]
            self.assertEqual({row["seed"] for row in rows}, {17 + replicate})
            self.assertEqual(
                {row["completed_tasks"] for row in rows},
                {row["task_count"] for row in rows},
            )
        with tempfile.TemporaryDirectory() as temporary:
            _, summary_path = run_experiments(
                config, Path(temporary) / "paired.csv", replicates=3,
                seed=17, duration_jitter=0.1,
            )
            with summary_path.open(newline="", encoding="utf-8") as summary_file:
                paired = [
                    row for row in csv.DictReader(summary_file)
                    if row["analysis"] == "paired_makespan_difference"
                ]
        sjf_difference = next(row for row in paired if row["policy"] == "sjf")
        self.assertEqual(float(sjf_difference["mean_makespan"]), 0)

    def test_loads_committed_repository_dag(self):
        tasks, dependencies = load_repository_workload(10, 0, data_size_per_edge=2)
        self.assertEqual(len(tasks), 10)
        self.assertTrue(dependencies)
        self.assertTrue(all(dep.data_size == 2 for dep in dependencies))
        with self.assertRaises(IndexError):
            load_repository_workload(10, 1000)
        held_out_tasks, held_out_edges = load_repository_workload(
            30, 0, split="test"
        )
        self.assertEqual(len(held_out_tasks), 30)
        self.assertTrue(held_out_edges)

    def test_runs_multiple_committed_workflows_as_paired_samples(self):
        config = Path("configs/edge_cloud_dataset.json")
        with tempfile.TemporaryDirectory() as temporary:
            rows, summary = run_experiments(
                config,
                Path(temporary) / "multi.csv",
                replicates=2,
                seed=31,
                dataset_size=10,
                dataset_indices=[0, 1, 2],
                data_size_per_edge=1,
            )
            self.assertEqual(len(rows), 24)
            for index in (0, 1, 2):
                by_policy = {
                    row["policy"]: row["makespan"]
                    for row in rows
                    if row["dataset_index"] == index and row["replicate"] == 0
                }
                self.assertEqual(
                    set(by_policy), {"fcfs", "sjf", "min_finish", "stale_min_finish"}
                )
            with summary.open(newline="", encoding="utf-8") as summary_file:
                self.assertEqual(len(list(csv.DictReader(summary_file))), 8)

    def test_sensitivity_runner_writes_labeled_paired_scenarios(self):
        config = Path("configs/edge_cloud_example.json")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sensitivity.csv"
            raw_output = Path(temporary) / "sensitivity_raw.csv"
            log_output = Path(temporary) / "sensitivity.log"
            rows = run_sensitivity(
                config,
                output,
                data_sizes=[0, 1],
                observation_delays=[0],
                link_scales=[1],
                node_capacity_scales=[1],
                bootstrap_samples=20,
                raw_output_path=raw_output,
                log_path=log_output,
            )
            self.assertEqual(len(rows), 16)
            with output.open(newline="", encoding="utf-8") as result_file:
                written = list(csv.DictReader(result_file))
            self.assertEqual(len(written), 16)
            with raw_output.open(newline="", encoding="utf-8") as raw_file:
                raw_rows = list(csv.DictReader(raw_file))
            self.assertEqual(len(raw_rows), 8)
            self.assertIn("scenario=1", log_output.read_text(encoding="utf-8"))
            self.assertEqual(
                {float(row["data_size_per_edge"]) for row in written}, {0, 1}
            )
            stale_comparisons = [
                row for row in written
                if row["comparison_policy"] == "min_finish"
            ]
            self.assertEqual(len(stale_comparisons), 2)
            report_path = Path(temporary) / "report.md"
            figure_dir = Path(temporary) / "figures"
            policy_plot, heatmap = generate_report(
                output, raw_output, report_path, figure_dir
            )
            self.assertIn("Sensitivity Analysis", report_path.read_text(encoding="utf-8"))
            self.assertTrue(policy_plot.exists())
            self.assertTrue(heatmap.exists())
            ET.parse(policy_plot)
            ET.parse(heatmap)


if __name__ == "__main__":
    unittest.main()
