"""Deterministic discrete-event simulator for Kubernetes-inspired Edge/Cloud DAG scheduling.

This is a simulator only; it does not interact with Kubernetes or replace clusterEnv.
Network transfer is conservatively modeled as a task release delay after each parent
finishes. Transfers do not contend for bandwidth in this first prototype.
"""

from dataclasses import dataclass, field
import heapq
import math
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Node:
    node_id: str
    tier: str
    cpu_capacity: float
    memory_capacity: float
    speed: float = 1.0

    def __post_init__(self):
        if not self.node_id:
            raise ValueError("node_id must not be empty")
        if self.tier not in {"edge", "cloud"}:
            raise ValueError("tier must be 'edge' or 'cloud'")
        if not all(math.isfinite(value) for value in (
            self.cpu_capacity, self.memory_capacity, self.speed
        )):
            raise ValueError("node capacities and speed must be finite")
        if self.cpu_capacity <= 0 or self.memory_capacity <= 0 or self.speed <= 0:
            raise ValueError("node capacities and speed must be positive")


@dataclass(frozen=True)
class Task:
    task_id: str
    duration: float
    cpu: float
    memory: float
    allowed_tiers: Tuple[str, ...] = ()

    def __post_init__(self):
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if not all(math.isfinite(value) for value in (self.duration, self.cpu, self.memory)):
            raise ValueError("task duration and resource requests must be finite")
        if self.duration < 0 or self.cpu <= 0 or self.memory <= 0:
            raise ValueError("task duration must be non-negative and resource requests positive")
        if any(tier not in {"edge", "cloud"} for tier in self.allowed_tiers):
            raise ValueError("allowed_tiers may contain only 'edge' or 'cloud'")


@dataclass(frozen=True)
class Dependency:
    parent: str
    child: str
    data_size: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.data_size) or self.data_size < 0:
            raise ValueError("dependency data_size must be non-negative")


@dataclass
class Pod:
    pod_id: str
    task_id: str
    cpu_request: float
    memory_request: float
    node_id: str
    status: str = "pending"


@dataclass(frozen=True)
class Link:
    latency: float
    bandwidth: float

    def __post_init__(self):
        if not math.isfinite(self.latency) or not math.isfinite(self.bandwidth):
            raise ValueError("latency and bandwidth must be finite")
        if self.latency < 0 or self.bandwidth <= 0:
            raise ValueError("latency must be non-negative and bandwidth positive")


@dataclass
class _NodeState:
    spec: Node
    available_cpu: float = field(init=False)
    available_memory: float = field(init=False)

    def __post_init__(self):
        self.available_cpu = self.spec.cpu_capacity
        self.available_memory = self.spec.memory_capacity


class EdgeCloudSimulator:
    """Runs a workflow using deterministic FCFS, SJF, or earliest-finish scheduling."""

    def __init__(
        self,
        nodes: Iterable[Node],
        tasks: Iterable[Task],
        dependencies: Iterable[Dependency],
        links: Optional[Dict[Tuple[str, str], Link]] = None,
        observation_delay: float = 0.0,
    ):
        if not math.isfinite(observation_delay) or observation_delay < 0:
            raise ValueError("observation_delay must be finite and non-negative")
        self.observation_delay = observation_delay
        node_list = list(nodes)
        task_list = list(tasks)
        dependency_list = list(dependencies)
        self._validate_unique(
            [(dep.parent, dep.child) for dep in dependency_list], "dependency"
        )
        self._validate_unique([n.node_id for n in node_list], "node")
        self._validate_unique([t.task_id for t in task_list], "task")
        if not node_list or not task_list:
            raise ValueError("at least one node and one task are required")
        self.nodes = {n.node_id: _NodeState(n) for n in node_list}
        self.tasks = {t.task_id: t for t in task_list}
        self._task_order = {task.task_id: index for index, task in enumerate(task_list)}
        self.dependencies = dependency_list
        self.links = dict(links or {})
        self.parents = {task_id: [] for task_id in self.tasks}
        self.children = {task_id: [] for task_id in self.tasks}
        for dep in self.dependencies:
            if dep.parent not in self.tasks or dep.child not in self.tasks:
                raise ValueError("dependency references an unknown task")
            if dep.parent == dep.child:
                raise ValueError("self-dependencies are not allowed")
            self.parents[dep.child].append(dep)
            self.children[dep.parent].append(dep)
        self._validate_acyclic()
        self._validate_links()
        self.time = 0.0
        self.completed = set()
        self.placements = {}
        self.pods = {}
        self.trace = []
        self._has_run = False
        self._events = []
        self._sequence = 0
        self._start_times = {}
        self._finish_times = {}
        self._task_ready_at = {task_id: 0.0 for task_id in self.tasks}
        self._busy_time = {node_id: 0.0 for node_id in self.nodes}
        self._cpu_busy_time = {node_id: 0.0 for node_id in self.nodes}
        self._memory_busy_time = {node_id: 0.0 for node_id in self.nodes}
        self._observed_cpu = {node_id: state.available_cpu for node_id, state in self.nodes.items()}
        self._observed_memory = {
            node_id: state.available_memory for node_id, state in self.nodes.items()
        }
        self._last_observation_time = 0.0
        self._stale_decisions = 0
        self._observation_wait_time = 0.0
        self.communication_time = 0.0
        self.transferred_bytes = 0.0

    @staticmethod
    def _validate_unique(values, label):
        if len(values) != len(set(values)):
            raise ValueError(f"{label} IDs must be unique")

    def _validate_acyclic(self):
        indegree = {task_id: len(self.parents[task_id]) for task_id in self.tasks}
        ready = [task_id for task_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for dep in self.children[current]:
                indegree[dep.child] -= 1
                if indegree[dep.child] == 0:
                    ready.append(dep.child)
        if visited != len(self.tasks):
            raise ValueError("workflow dependencies must form a DAG")

    def _validate_links(self):
        known_nodes = set(self.nodes)
        for pair, link in self.links.items():
            if len(pair) != 2 or pair[0] not in known_nodes or pair[1] not in known_nodes:
                raise ValueError("network link references an unknown node")
            if not isinstance(link, Link):
                raise TypeError("link values must be Link instances")

    def _transfer_delay(self, source, destination, data_size):
        if source == destination or data_size == 0:
            return 0.0
        link = self.links.get((source, destination))
        if link is None:
            link = self.links.get((destination, source))
        if link is None:
            raise ValueError(f"missing network link between {source} and {destination}")
        return link.latency + data_size / link.bandwidth

    def _feasible_nodes(self, task):
        return [
            node_id for node_id, state in self.nodes.items()
            if state.available_cpu >= task.cpu
            and state.available_memory >= task.memory
            and (not task.allowed_tiers or state.spec.tier in task.allowed_tiers)
        ]

    def _refresh_observation(self):
        if self.time - self._last_observation_time >= self.observation_delay:
            self._observed_cpu = {
                node_id: state.available_cpu for node_id, state in self.nodes.items()
            }
            self._observed_memory = {
                node_id: state.available_memory for node_id, state in self.nodes.items()
            }
            self._last_observation_time = self.time

    def _advance(self):
        completion_time = self._events[0][0] if self._events else math.inf
        transfer_times = [
            self._task_ready_at[task_id]
            for task_id in self.tasks
            if task_id not in self.placements
            and self.parents[task_id]
            and all(dep.parent in self.completed for dep in self.parents[task_id])
            and self._task_ready_at[task_id] > self.time
        ]
        transfer_time = min(transfer_times, default=math.inf)
        observation_time = (
            self._last_observation_time + self.observation_delay
            if self.observation_delay > 0
            and self._last_observation_time + self.observation_delay > self.time
            else math.inf
        )
        next_time = min(completion_time, transfer_time, observation_time)
        if math.isinf(next_time):
            raise RuntimeError("no feasible task and no pending event: workflow is unschedulable")
        if completion_time <= transfer_time and completion_time <= observation_time:
            finish, _, task_id, node_id = heapq.heappop(self._events)
            self.time = finish
            task = self.tasks[task_id]
            node = self.nodes[node_id]
            node.available_cpu += task.cpu
            node.available_memory += task.memory
            self.completed.add(task_id)
            self._finish_times[task_id] = finish
            self.pods[task_id].status = "completed"
            self.trace.append({
                "time": finish, "event": "completed", "task_id": task_id,
                "node_id": node_id,
            })
            self._refresh_observation()
            return
        self.time = next_time
        if observation_time <= transfer_time:
            self._refresh_observation()
            self.trace.append({"time": self.time, "event": "observation_refreshed"})
        else:
            self.trace.append({"time": self.time, "event": "transfer_ready"})

    def run(self, policy="fcfs"):
        if self._has_run:
            raise RuntimeError("simulator instances are single-use; create a new instance")
        if policy not in {"fcfs", "sjf", "min_finish", "stale_min_finish"}:
            raise ValueError(
                "policy must be 'fcfs', 'sjf', 'min_finish', or 'stale_min_finish'"
            )
        self._has_run = True
        while len(self.completed) < len(self.tasks):
            ready = [
                task for task_id, task in self.tasks.items()
                if task_id not in self.placements
                and all(dep.parent in self.completed for dep in self.parents[task_id])
                and self._task_ready_at[task_id] <= self.time
            ]
            feasible = []
            physically_feasible = False
            for task in ready:
                for node_id in self._feasible_nodes(task):
                    physically_feasible = True
                    observed = (
                        self._observed_cpu[node_id] >= task.cpu
                        and self._observed_memory[node_id] >= task.memory
                    )
                    if observed:
                        feasible.append((task, node_id))
            if not feasible:
                if physically_feasible and self.observation_delay > 0:
                    self._stale_decisions += 1
                # The scheduler cannot place from its cached view. Wait for a state change.
                previous_time = self.time
                self._advance()
                if physically_feasible and self.time > previous_time:
                    self._observation_wait_time += self.time - previous_time
                continue
            if self.observation_delay > 0 and self.time > self._last_observation_time:
                self._stale_decisions += 1
            if not feasible:
                self._advance()
                continue
            if policy in {"min_finish", "stale_min_finish"}:
                def estimated_finish(pair):
                    task, node_id = pair
                    transfer_ready = max(
                        (
                            self._finish_times[dep.parent]
                            + self._transfer_delay(
                                self.placements[dep.parent], node_id, dep.data_size
                            )
                            for dep in self.parents[task.task_id]
                        ),
                        default=self.time,
                    )
                    return (
                        max(self.time, transfer_ready)
                        + task.duration / self.nodes[node_id].spec.speed,
                        self._task_order[task.task_id],
                        node_id,
                    )

                feasible.sort(key=estimated_finish)
            elif policy == "sjf":
                feasible.sort(key=lambda pair: (
                    pair[0].duration,
                    self._task_order[pair[0].task_id],
                    pair[1],
                ))
            else:
                feasible.sort(key=lambda pair: (self._task_order[pair[0].task_id], pair[1]))
            task, node_id = feasible[0]
            state = self.nodes[node_id]
            state.available_cpu -= task.cpu
            state.available_memory -= task.memory
            self._observed_cpu[node_id] -= task.cpu
            self._observed_memory[node_id] -= task.memory
            parent_releases = []
            for dep in self.parents[task.task_id]:
                delay = self._transfer_delay(
                    self.placements[dep.parent], node_id, dep.data_size
                )
                self.communication_time += delay
                if delay:
                    self.transferred_bytes += dep.data_size
                parent_releases.append(self._finish_times[dep.parent] + delay)
            self._task_ready_at[task.task_id] = max(parent_releases, default=0.0)
            self.placements[task.task_id] = node_id
            pod = Pod(
                f"pod-{task.task_id}", task.task_id, task.cpu, task.memory,
                node_id=node_id, status="running",
            )
            self.pods[task.task_id] = pod
            start = max(self.time, self._task_ready_at[task.task_id])
            finish = start + task.duration / self.nodes[node_id].spec.speed
            self._start_times[task.task_id] = start
            runtime = finish - start
            self._busy_time[node_id] += runtime
            self._cpu_busy_time[node_id] += task.cpu * runtime
            self._memory_busy_time[node_id] += task.memory * runtime
            self._sequence += 1
            heapq.heappush(self._events, (finish, self._sequence, task.task_id, node_id))
            self.trace.append({
                "time": self.time, "event": "scheduled", "task_id": task.task_id,
                "pod_id": pod.pod_id, "node_id": node_id, "start": start, "finish": finish,
            })
        return self.metrics()

    def metrics(self):
        makespan = max(self._finish_times.values(), default=0.0)
        return {
            "makespan": makespan,
            "completed_tasks": len(self.completed),
            "task_count": len(self.tasks),
            "communication_delay": self.communication_time,
            "transferred_bytes": self.transferred_bytes,
            "stale_observation_decisions": self._stale_decisions,
            "observation_delay": self.observation_delay,
            "observation_wait_time": self._observation_wait_time,
            "node_utilization": {
                node_id: {
                    "cpu": self._cpu_busy_time[node_id] / (state.spec.cpu_capacity * makespan)
                    if makespan else 0.0,
                    "memory": self._memory_busy_time[node_id] / (state.spec.memory_capacity * makespan)
                    if makespan else 0.0,
                }
                for node_id, state in self.nodes.items()
            },
            "task_start_times": dict(self._start_times),
            "task_finish_times": dict(self._finish_times),
            "placements": dict(self.placements),
            "pod_status": {task_id: pod.status for task_id, pod in self.pods.items()},
        }
