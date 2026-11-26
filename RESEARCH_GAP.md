# Research Gap

## Evidence boundary

This document records what is supported by repository source and what remains a research proposal. A feature shown as “missing” was not found in the inspected implementation; it is not inferred from a paper or external benchmark. The existing source is a single-resource DAG workflow simulator. No Edge–Cloud co-scheduling implementation exists in the checked-out commit.

## Original research problem

The README presents the original project as cloud workflow scheduling using deep reinforcement learning. Tasks form a DAG; each task has a duration and CPU/memory demand; precedence relationships constrain eligibility; and the simulator's principal visible objective is workflow makespan. The source's principal environment, `Envs/clusterEnv.py`, models one fixed-capacity CPU/memory resource pool and schedules ready tasks onto that pool.

## Existing capabilities and gaps

| Research requirement | Existing implementation | Missing functionality | Required modification | Difficulty | Scientific relevance |
|---|---|---|---|---|---|
| DAG task dependencies | `edges` lists, `"Start"`/`"Exit"` sentinels, and ready-list logic in `clusterEnv` | No verified inter-application dependencies or transfer-size semantics on edges | Preserve DAG model; add typed edge metadata if transfer cost is studied | Medium | Essential application workflow constraint |
| CPU and memory availability | Fixed pool capacity 100/100; per-task demand and feasibility checks | Per-node capacities, reservations across cluster, and resource overcommit policy | Introduce node resource ledgers and feasibility checks | Medium | Required to model resource allocation |
| Heterogeneous compute | All tasks share one abstract capacity pool | Different CPU speed, memory size, accelerators, or performance profiles | Define node capacity/performance vectors and normalized execution time | Medium | Central to heterogeneous scheduling |
| Edge–Cloud tiers | Not present | Edge nodes, cloud nodes, tier properties, and placement trade-offs | Add explicit node tiers and tier-specific parameters | Medium | Directly addresses research context |
| Kubernetes pods | Not present | Pod requests, limits, lifecycle, and node-selector-like constraints | Add a Kubernetes-inspired pod abstraction in the simulator | Medium | Needed to reason about pod-level placement |
| Task-to-pod assignment | Not present; task is submitted directly to one pool | Mapping of DAG tasks to pods or pod templates | Define task resource profile and assignment rule | Medium | Enables application/pod co-scheduling |
| Pod-to-node placement | Not present | Node eligibility, placement, and reservation | Add feasible-node selection and resource reservation | Medium | Enables cluster scheduling |
| Joint co-scheduling | Not present | Coupled task->pod and pod->node decisions | Implement one coordinated decision procedure and comparable baselines | High | Primary target of the master's work |
| Network communication cost | No executable network model found | Latency/bandwidth between nodes; data size per DAG edge | Add link matrix and explicit transfer-delay calculation | Medium | Distinguishes Edge–Cloud from single-cluster scheduling |
| Server workload/load balance | Single-pool remaining resources only | Node-level utilization and imbalance objective | Track per-node utilization and define an explicit balance measure | Medium | Relevant secondary scheduling objective |
| Priorities and criticality | No priority/deadline model confirmed | Deadline, critical-path urgency, and task classes | Add task metadata and a documented priority policy | Medium | Supports differentiated application QoS |
| Partial/stale information | None in original | Delayed, noisy, or missing cluster telemetry | Prototype has fixed-delay stale free-capacity snapshots; noise/missing data remain absent | High | Directly motivates robust scheduling |
| RL | PPO and Actor-Critic scripts, plus a graph environment | Reproducible, working RL execution and action/state model for multi-node co-scheduling | First validate baseline; then define a new action/state and reward for the new simulator | High | Potential method, not prerequisite or proven advantage |
| Baseline algorithms | Random, SJF, Tetris-like, MCTS code in separate scripts | Verified, fair, consistently configured experiments | Build shared evaluation harness and comparable constraints | Medium | Necessary to assess any claimed benefit |
| Performance evaluation | Makespan output and stored spreadsheet/plot files | Repeated seeds, raw CSV, uncertainty summaries, full provenance | Add reproducible evaluation configuration and artifact generation | Medium | Necessary for scientific claims |
| Real Kubernetes integration | None found | Kubernetes control-plane interaction and cluster deployment | Separate future deployment project; not needed for simulator contribution | Very high | Only relevant if real-system validation is claimed |

Difficulty is a preliminary engineering estimate, not a measured schedule.

## Technical limitations of the original

1. The main environment has one fixed CPU/memory pool rather than multiple machines.
2. The source does not represent pods, node placement, edge/cloud topology, network latency, deadlines, priority, or monitoring staleness.
3. Some environment variants import `Envs.utils`, which is absent; the main `clusterEnv` was made importable by removing its unused import, but other variants remain unresolved.
4. Several environment variants contain absolute paths from the original author's macOS workstation.
5. Root scripts call custom Gym IDs without registering them in tracked source.
6. The code targets a legacy Gym step interface and lacks an explicit dependency manifest.
7. Terminal completion previously returned a stale makespan observation; this was corrected in `clusterEnv`. The last completion reward still returns zero, so RL return accounting is incomplete.
8. The generator's active CPU/memory demand distribution and duration probability differ from the README description (details in `SYSTEM_ARCHITECTURE.md`).
9. TensorBoard and cluster environment registration are available. An eight-transition PPO update and one checkpoint inference episode passed, but full PPO training and statistical evaluation have not been performed. The final completion reward omits the last simulated-time interval; full training is postponed while preserving the original baseline behavior.
10. Existing spreadsheets, models, figures, and event logs have no confirmed provenance in the repository and have not been reproduced.

## Scientific gap

The source demonstrates DAG scheduling with two resource dimensions on a single simulated compute pool. The master's problem couples application-level DAG task readiness with pod-level resource requirements and node-level placement across heterogeneous Edge and Cloud resources, while data movement and imperfect monitoring affect feasibility and completion time.

The principal gap is therefore not “add a more advanced RL algorithm” to the old environment. The missing system model and decision coupling must first be represented. Without pod, node, network, and observation-age models, evaluating a policy on the current environment would not test the stated research question.

## Proposed contribution and hypotheses

### Engineering extension (simulator prototype)

Build a reproducible discrete-event simulator that:

- represents Edge and Cloud nodes with heterogeneous CPU/memory capacities and processing rates;
- models applications as DAGs whose edges can carry data size;
- adds Kubernetes-inspired pods with explicit resource requests and task-to-pod membership;
- schedules task-to-pod assignment and pod-to-node placement as a coupled decision;
- models network transfer delay between nodes;
- optionally exposes delayed node-capacity observations (noise/missing data not modeled);
- evaluates makespan, end-to-end latency, resource utilization, load imbalance, and deadline violations where deadlines are modeled.

This describes a proposed simulator, not a real Kubernetes scheduler. A real-cluster integration is explicitly outside the first prototype.

### Research hypotheses (not established contributions)

- **H1:** Joint decisions can reduce a selected application-level objective compared with a decoupled policy under the same workload and resource assumptions.
- **H2:** Explicit transfer delay changes placement decisions and can change workflow completion time relative to a zero-network-cost model.
- **H3:** A resource-aware heuristic provides a useful interpretable baseline before considering a learned policy.
- **H4:** Decision quality may degrade under stale information; the effect can be measured by varying observation delay.

These are testable hypotheses, not novelty claims or expected results. A literature review and controlled experiments are required before asserting novelty or support.

## Proposed modifications and staged difficulty

| Stage | Addition | Difficulty | Validation before next stage |
|---|---|---|---|
| 0 | Finish documenting and stabilizing the original baseline | Low–Medium | Reproducible smoke/full episode; agreed reward semantics |
| 1 | Heterogeneous node model and resource ledger | Medium | Unit tests for capacity, reservation, and release |
| 2 | Edge/Cloud node tiers and processing rates | Low–Medium | Deterministic placement scenarios |
| 3 | DAG workload model with successor/precedence tests | Medium | Verify all generated topological constraints |
| 4 | Pod abstraction and resource requests | Medium | Pod admission and fit tests |
| 5 | Task-to-pod plus pod-to-node joint heuristic | High | Compare against separately defined simple baselines |
| 6 | Network latency/data transfer on DAG edges | Medium | Hand-calculated transfer-delay scenarios |
| 7 | Reproducible metric collection and experiment runner | Medium | CSV schema, seeds, repeated runs, output validation |
| 8 | Delayed/stale observation model | Medium–High | Delay ablation and consistency checks |
| 9 | RL policy only if justified by baseline performance and research question | High | State/action/reward tests, training stability, repeated comparisons |
| 10 | Real Kubernetes validation, if required | Very High | Separate deployment plan and approval |

## Scientific justification and limitations

Beginning with a simulator and a transparent heuristic limits engineering risk and makes each modeled assumption inspectable. It allows controlled experiments for resource heterogeneity and communication costs before RL adds training complexity. The simulator cannot by itself establish behavior in real Kubernetes: it abstracts container startup, scheduler plugins, control-plane latency, runtime contention, and noisy hardware performance.

The original repository should remain the historical baseline. The new simulator should be an isolated extension rather than a silent rewrite of the original algorithm or experiment artifacts. Comparisons are meaningful only when input workloads, resource constraints, arrival model, task runtime assumptions, and metric definitions are aligned.

## Current status

- **Original DAG simulator:** direct reset/step and single-episode smoke-tested after documented non-scheduling fixes.
- **SJF helper:** one complete workflow smoke-tested; not statistically evaluated.
- **PPO/GCN/MCTS:** present in source in varying forms; not reproduced on this workstation.
- **Edge–Cloud/pod/network co-scheduler:** isolated simulator prototype implemented, including fixed-delay stale free-capacity observations and a task-scoped pod lifecycle. No live Kubernetes integration, reusable pods, or measured network/telemetry data.
- **Research novelty/results:** not established.
