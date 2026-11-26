# Repository Analysis

## Scope and evidence

This is an initial source-level analysis of the checked-out repository. It distinguishes repository documentation and code from files that merely happen to be stored in the repository. The scheduling decision logic has not been changed. A direct environment test and one SJF episode now pass after narrow execution/measurement fixes; the single-run values below are smoke-test observations, not experimental conclusions.

- Repository: `https://github.com/maissayahyaoui3-source/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning`
- Checked-out commit: `eb246ebba160567277c9c1aa226e359f48629dac` (2022-04-21)
- Current branch: `feat/edge-cloud-co-scheduling-prototype`
- Main language: Python
- License: MIT (`LICENSE`)
- Repository has no `requirements.txt`, `pyproject.toml`, `setup.py`, or automated test suite. A small `Envs/registration.py` compatibility helper has since been added for `clusterEnv-v0`; the other environment IDs are not registered.
- A pre-existing untracked `.idea/` directory was present during inspection and was not modified.

## Repository structure

| Path | Observed contents and purpose |
|---|---|
| `README.md` | Chinese-language project description, old setup instructions, DAG model, state/action/reward descriptions, and summary of experiments. |
| `Envs/` | Four custom Gym-style environments: `clusterEnv.py`, `myEnv.py`, `graphEnv.py`, and `testEnv.py`; `registration.py` registers `clusterEnv-v0` for cluster-based scripts. The referenced `Envs/utils.py` is absent. |
| `DAGs_Generator.py` | Random layered-DAG and task duration/resource-demand generator; also contains train/test dataset export functions. |
| `DRLagent.py`, `DRL_test.py` | Actor-Critic training and evaluation path for `MyEnv-v0`. |
| `PPOagent.py`, `PPOtest.py` | PPO training and evaluation path for `clusterEnv-v0`. |
| `PPOagentForGCN.py`, `PPOtestForGCN.py` | PPO variant using the graph environment and GCN-derived state representation. |
| `MonteCarloTreeSearch.py`, `AblationExperiment.py`, `resources_monitor MCTS.py` | MCTS-based or MCTS-related experiment scripts. The first includes PPO-related classes as well as a tree search; exact experiment provenance and reproducibility remain unverified. |
| `baselines_tableau.py`, `resources_monitor_tetris.py`, `resources_monitor.py` | Baseline evaluation and resource-monitoring scripts. `baselines_tableau.py` defines Random, Shortest Job First (SJF), and Tetris-like choices. |
| `GCNutils/` | Graph-convolution layer, GCN model, and graph-data helpers. |
| `GCN_initialization/` | Serialized initial weights used by `Envs/graphEnv.py`. |
| `npy/train_datasheet/`, `npy/test_datasheet/` | Pre-generated DAG edge, duration, and demand arrays for sizes 10, 20, 30, 40, and 50; an additional 100-workflow test collection for 30-task DAGs is present under `test/`. |
| `runs/PPO_preTrained/`, `models/`, `runs/` | Model/checkpoint files and TensorBoard event files. Their provenance, producing command, and associated code version are not established by the repository. |
| `data/`, `figure/` | Existing spreadsheet and plot artifacts, including makespan-labelled files. Their values have not been independently reproduced. |
| `README.assets/` | Diagrams and figures embedded in the README. |
| `.vscode/` | Editor extension recommendation and a generic Python current-file launch configuration. |
| `normalization.py` | Normalization helper; its role in the main workflow is not established by current entry points. |
| `res_monitor.xls`, `data/res_monitor.xls` | Existing spreadsheet artifacts. |

The tree also contains `.DS_Store` files and tracked `__pycache__` bytecode. There is no independent documentation directory in the checked-out source.

## Project purpose and original research problem

The README describes a bachelor's thesis project titled *基于深度强化学习的云工作流调度* (“Cloud Workflow Scheduling Based on Deep Reinforcement Learning”). It models a workflow as a directed acyclic graph (DAG), with precedence constraints between tasks, and schedules tasks on a simulated compute resource with CPU and memory capacities. The README says the objective is to reduce workflow makespan, and discusses PPO/Actor-Critic, graph representations, Monte Carlo Tree Search (MCTS), and baselines.

The actual environment source confirms that the principal simulation is not a Kubernetes cluster: `Envs/clusterEnv.py` models one compute resource with fixed capacity 100 CPU units and 100 memory units. Tasks that fit are allowed to run concurrently, and the simulator advances its clock when the shortest remaining task completes. The DAG captures task dependencies, while task duration and CPU/memory demand are loaded from generated arrays. The implementation does not model nodes, pods, containers, network latency, or Edge–Cloud placement.

The README mentions Tetris and Decima and links to related work; the repository does not include a paper or a reproducible mapping from each stored artifact to an exact experiment.

## Main modules and responsibilities

| Module | Responsibility confirmed from source |
|---|---|
| `DAGs_Generator.py` | `DAGs_generate()` creates layered DAG edges, and `workflows_generator()` adds task durations and CPU/memory demands. `generate_train_datasheet()` and `generate_test_datasheet()` write `.npy` datasets. |
| `Envs/clusterEnv.py` | `clusterEnv` implements the main fixed-capacity, multi-resource workflow simulator, with ready-task selection and a Gym-style `reset()` / `step()` interface. It supplies `return_dim_info()` and state accessors used by the MCTS code. |
| `Envs/myEnv.py` | Similar fixed-resource DAG scheduling environment used by the Actor-Critic path. |
| `Envs/graphEnv.py` | GCN/graph-embedding environment and `DecimaGNN`; loads GCN initialization weights. |
| `Envs/testEnv.py` | Another environment variant with a 10-task configuration. |
| `PPOagent.py` | PPO Actor-Critic implementation and `train()` loop; defaults to `clusterEnv-v0`, 300,000 training timesteps, and checkpoint output under `runs/PPO_preTrained/`. |
| `PPOtest.py` | Loads a selected `clusterEnv-v0` PPO checkpoint, evaluates 1,000 episodes in groups of ten, and writes makespans to `data/makespan_PPO.xls`. |
| `PPOagentForGCN.py`, `PPOtestForGCN.py` | PPO training/testing scripts for `graphEnv-v0`. |
| `DRLagent.py`, `DRL_test.py` | Separate Actor-Critic training/testing path for `MyEnv-v0`. |
| `baselines_tableau.py` | Defines Random, SJF, and Tetris-like scheduling policies and writes spreadsheet results. |
| `MonteCarloTreeSearch.py`, `AblationExperiment.py` | Tree search over environment actions, with different MCTS experiment implementations. |
| `GCNutils/layers.py`, `models.py`, `utils.py` | GCN layers, model, and graph conversion helpers. |
| `resources_monitor*.py` | Experiment variants labelled for monitoring; portions of the resource logging/output are commented out. |

## Implemented scheduling algorithms

The codebase contains several separate experiment paths rather than one consistently packaged application:

- **PPO**: `PPOagent.py` implements an Actor-Critic policy with clipped PPO updates for `clusterEnv-v0`; the graph variant is in `PPOagentForGCN.py`.
- **Actor-Critic / REINFORCE-style training**: `DRLagent.py` has separate policy and value networks and a `trainIters()` training function for `MyEnv-v0`.
- **MCTS**: tree search is present in `MonteCarloTreeSearch.py` and `AblationExperiment.py`; there are related resource-monitoring scripts.
- **Heuristic/random baselines**: `baselines_tableau.py` defines Random, SJF, and Tetris-like task selection.

The README discusses a DRL-guided MCTS idea, but the code paths should be inspected and executed before claiming that every script implements the same hybrid algorithm or reproduces the reported method. No algorithm has been run in the current environment.

## Scheduling model and expected outputs

In the main `clusterEnv` design, the state contains elapsed simulated time, remaining CPU and memory, a bounded list of ready-task durations and demands, and additional DAG/backlog features. The action chooses a ready task to submit or advances execution by completing the shortest-running pending task. Resource feasibility is checked against the single simulated resource. The environment returns negative elapsed-time increments as reward when time advances; task submission gives zero reward.

The principal metric visible in the code and saved artifact names is **makespan**. The PPO training script also writes episode makespan and cumulative reward to TensorBoard. Evaluation scripts attempt to write Excel workbooks. A reliable evaluation protocol, complete metric set, and exact correspondence between saved spreadsheets/figures and code versions are not documented.

## Dependencies and runtime

`README.md` explicitly requests Python 3.9.7, PyTorch 1.10, Gym 0.21.0, and NetworkX 2.6.3. Source imports additionally show requirements for NumPy, SciPy, Matplotlib, TensorBoard, `xlwt`, and `xlsxwriter`. No dependency manifest pins these additional packages or resolves their compatibility.

The inspected machine has Python 3.12.0 and globally available PyTorch 2.11.0+cpu, NumPy 2.4.4, NetworkX 3.6.1, and SciPy 1.17.1. Gym, Matplotlib, TensorBoard, `xlwt`, and `xlsxwriter` were initially missing. With approval to try the Python 3.12 compatibility route, a venv was created at `%LOCALAPPDATA%\venvs\cwscheduling-py312` using `--system-site-packages`; Gym 0.26.2, `xlwt` 1.3.0, and `xlsxwriter` 3.3.4 were installed in it. The venv therefore shares the globally installed numerical/ML packages and is a compatibility probe, not a fully locked or isolated reproduction environment.

Gym 0.26.2 emits its warning that it is unmaintained and does not support NumPy 2.x; the current venv sees global NumPy 2.4.4. The original source uses the older four-return `step()` API. A direct class smoke test passed, and the new `Envs/registration.py` helper disables Gym's API checker for `clusterEnv-v0` on Gym 0.25+; a complete episode through the registered environment also passed. This does not establish full PPO compatibility. Matplotlib installation timed out during download. TensorBoard is still needed for PPO and an installation attempt is in progress/has not been verified. `pip check` previously reported no broken requirements in the shared venv.

## Dataset availability

The repository includes train and test arrays for 1,000 DAGs each at sizes 10, 20, 30, 40, and 50. For each size, observed array shapes are:

- `edges*_lib.npy`: `(1000,)`, dtype `object`
- `duration*_lib.npy`: `(1000, N)`, dtype `float32`
- `demand*_lib.npy`: `(1000, N, 2)`, dtype `float32`

An additional test set at `npy/test_datasheet/test/` contains 100 samples with 30 tasks each. These arrays were inspected locally; no external dataset download is indicated. The edge arrays are NumPy object arrays and source loads them with `allow_pickle=True`.

## Main execution and evaluation entry points

The README describes a workflow of dataset generation, PPO training/testing, baseline evaluation, and MCTS evaluation. Actual root-level files are:

1. `DAGs_Generator.py` — dataset generator (its current `__main__` calls `generate_test_datasheet(1000, 100)`).
2. `PPOagent.py` — PPO training entry point.
3. `PPOtest.py` — PPO evaluation entry point.
4. `PPOagentForGCN.py` and `PPOtestForGCN.py` — GCN/PPO variants.
5. `baselines_tableau.py` — Random/SJF/Tetris-like baseline evaluation.
6. `MonteCarloTreeSearch.py` and `AblationExperiment.py` — MCTS experiment entry points.

At present, these are **not all verified executable commands**. The README's paths differ from the checked-out tree, other environment classes still contain hard-coded paths from the original author's macOS machine, `MyEnv-v0` and `graphEnv-v0` remain unregistered, and other environment modules still reference the absent `Envs.utils` module. The main `clusterEnv` class has received limited execution/measurement fixes and `clusterEnv-v0` now has a compatibility registration helper. TensorBoard and a short PPO update were verified, but full script and experiment outputs are not.

## Current execution attempt and potential problems

The initial environment checks were performed before code changes. The final direct smoke test was performed after the minimal `clusterEnv.py` portability fixes:

| Check | Result |
|---|---|
| `python --version` | Python 3.12.0 |
| Import installed numerical packages | PyTorch 2.11.0+cpu, NumPy 2.4.4, NetworkX 3.6.1, and SciPy 1.17.1 imported successfully. |
| Global `python -c "import gym"` | Failed: `ModuleNotFoundError: No module named 'gym'`. |
| Initial venv import of `Envs.clusterEnv` | Failed because the module imported absent `Envs.utils`. |
| Venv direct environment smoke test after minimal fixes | Passed: loaded 1,000 size-30 workflows; `reset()` returned a `(98,)` observation; one valid action returned the legacy four-value step result, reward `0.0`, and `done=False`. |

Additional source-level risks identified:

1. `Envs/clusterEnv.py` dataset paths were changed from the original author's macOS absolute paths to paths relative to the source file. Other environment files still contain hard-coded macOS paths.
2. The missing `.utils` import was removed from `Envs/clusterEnv.py` after confirming its only remaining reference was commented-out code. Other environment modules still import or call the missing utility module.
3. `_check_episode_finish()` in `Envs/clusterEnv.py` previously fell through to `None` when an episode was not finished; it now explicitly returns `False`. This preserves the intended false/true semantics while making the result a Boolean.
4. On final task completion, `step()` previously returned an observation whose time was one task duration behind the simulator's internal clock. After a full-episode test exposed this discrepancy, the terminal observation time is now set to `self.time`. This does not change action selection or simulated execution, but corrects the makespan exposed to evaluation code. The final-completion reward still returns `0` rather than the negative time increment; this affects RL return accounting and remains unresolved to avoid changing the learning objective without approval.
5. Scripts call `gym.make("clusterEnv-v0")`, `gym.make("MyEnv-v0")`, or `gym.make("graphEnv-v0")`. `Envs/registration.py` now registers `clusterEnv-v0` for the relevant scripts, and a complete episode through `gym.make()` passed. The other IDs are still unregistered; PPO and full baseline entry points remain unverified.
6. The generator's current script entry point requests 100-task datasets, while committed pre-generated datasets only cover task counts 10–50 and the generation functions do not create output directories. This is not a safe command to run as-is.
7. Gym 0.26.2 warns about NumPy 2.x; the full stack has not been validated. Matplotlib download timed out and a separate TensorBoard download stalled and was cancelled. Training and generator plotting are not ready.
8. Several scripts write fixed output filenames and can overwrite existing experiment artifacts when they run.
9. There are no automated tests or experiment configuration files, and random seeds are not consistently controlled.
10. Existing model, spreadsheet, plot, and TensorBoard files are present, but their provenance and reproducibility have not been confirmed.

### Executed smoke episodes

The direct original environment test completed a seeded size-30 DAG using the first valid ready task whenever possible:

- Seed: Python `random.seed(12345)` (selects the workflow from the committed array; it does not seed every library or constitute multi-run reproducibility).
- Completed tasks: 30/30.
- Decisions: 60.
- Simulator makespan: 234.0 time units.
- Workload duration sum: 453.0 time units.
- The pre-fix terminal observation reported 219.0 while `env.time` was 234.0. The terminal observation was corrected and the same episode was rerun successfully with terminal observation time 234.0.

The repository's actual SJF helper functions in `baselines_tableau.py` were then used for one episode with the same Python random seed:

- Completed tasks: 30/30.
- Decisions: 60.
- Makespan: 245.0 time units.

These are two single-run functional smoke observations, not a statistically valid comparison: workflow selection is random, no repeated runs or confidence intervals were performed, and the first policy is only a valid-action smoke policy rather than a named repository baseline.

## Preliminary relevance to the master's research

| Requirement | Existing evidence | Gap |
|---|---|---|
| DAG task scheduling | Implemented in custom simulation environments. | No multi-application workflow orchestration established. |
| CPU and memory constraints | Fixed CPU/memory capacities and per-task demands are modeled. | Only one abstract resource pool; no node-level heterogeneous capacities. |
| Heuristics and RL | PPO, Actor-Critic, MCTS, Random, SJF, and Tetris-like code exists. | Scripts are unintegrated, unvalidated here, and not a fair/reproducible comparison yet. |
| Edge–Cloud architecture | None confirmed. | Edge/Cloud nodes, tiers, and placement need to be modeled. |
| Kubernetes pods and pod placement | None confirmed. | No Kubernetes API, pod abstraction, or node placement. |
| Joint pod/task co-scheduling | None confirmed. | Requires a new model linking application tasks, pods, and nodes. |
| Network latency and communication cost | No such model confirmed. | DAG edges do not appear to carry executable communication-delay costs. |
| Partial/outdated cluster information | None confirmed. | No monitoring delay, uncertainty, or stale-state mechanism. |
| Evaluation | Makespan-oriented scripts and stored spreadsheet/plot artifacts. | Need reproducible runs, documented seeds/configuration, and validated metrics. |

The repository is potentially useful as a **DAG scheduling and algorithm baseline**, but not as an existing Edge–Cloud/Kubernetes co-scheduler. Its suitability depends on reproducing its original single-resource behavior. The proposed research model should remain a separate extension; no scheduling algorithm was changed in this phase.

## Precise execution plan

1. **Keep the chosen Python 3.12 compatibility route provisional.** The current external venv is usable for a direct smoke test but shares globally installed packages and has the Gym/NumPy compatibility warning.
2. **Resolve remaining required packages** without changing scheduling logic; check dependency versions and the old Gym API before PPO/baseline entry points. `clusterEnv-v0` registration is now implemented and smoke-tested.
3. **Preserve the original terminal reward and postpone full training for now.** TensorBoard is installed; an eight-transition PPO update and one checkpoint inference episode passed as smoke checks, but the terminal reward omits the final time interval and no full training/evaluation run is claimed.
4. **Compare observed behavior to README claims and existing artifacts** without treating stored output files as reproduced results.
5. **Keep the original simulator and algorithm intact** while documenting their limits; request approval before any research-architecture or scheduling-logic changes.

## Status

- Repository inspection: complete for this initial phase.
- Scheduling algorithm modifications: none.
- Environment setup: Python 3.12 venv compatibility probe prepared outside the repository; it shares preinstalled system-site packages and is not yet a reproducible locked environment.
- Original environment smoke test: passed for one `reset()` and one valid `step()` after minimal non-scheduling fixes in `Envs/clusterEnv.py`.
- One full 30-task environment episode and one SJF baseline-helper episode: passed as smoke runs; no repeated experimental evaluation.
- PPO training and evaluation: not run; TensorBoard installation and dependency compatibility remain unresolved. Registration of `MyEnv-v0` and `graphEnv-v0` also remains unresolved.
- Documentation created: `REPOSITORY_ANALYSIS.md`, `PROJECT_OVERVIEW.md`, `MANUAL_EXECUTION_GUIDE.md`, `INSTALLATION_GUIDE.md`, `EXECUTION_GUIDE.md`, and `CHANGELOG.md`.
