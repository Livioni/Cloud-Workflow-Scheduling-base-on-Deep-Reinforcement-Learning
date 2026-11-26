# Original Execution Guide

## Scope and evidence status

This guide separates source-level instructions from commands that have actually run on the Windows workstation. The original project is a simulator for DAG workflow tasks sharing one fixed CPU/memory pool. It is not a real Kubernetes or Edge–Cloud cluster.

The checked-out commit is `eb246ebba160567277c9c1aa226e359f48629dac`. The repository README describes dataset generation, PPO/DRL training, policy testing, baseline evaluation, and MCTS evaluation. The source originally lacked custom Gym environment registration; a small compatibility helper now registers `clusterEnv-v0` for the cluster-based scripts.

## Files and execution flow

| Purpose | File |
|---|---|
| Random DAG and task-workload generation | `DAGs_Generator.py` |
| Main fixed-resource simulator | `Envs/clusterEnv.py`, class `clusterEnv` |
| PPO training | `PPOagent.py`, function `train()` |
| PPO test/evaluation | `PPOtest.py`, function `test()` |
| Random/SJF/Tetris-like baseline implementations | `baselines_tableau.py` |
| MCTS experiment | `MonteCarloTreeSearch.py` |
| Graph/GCN PPO alternatives | `PPOagentForGCN.py`, `PPOtestForGCN.py`, `Envs/graphEnv.py` |

The core environment loads edge, duration, and demand arrays from `npy/train_datasheet/30/` in its current configuration. It exposes a 98-element observation and 31 discrete action indices through `return_dim_info()`. Its older Gym interface returns `(observation, reward, done, info)` from `step()`.

## Environment and source corrections made for execution

No scheduling decision algorithm was changed. The following non-algorithm execution/measurement corrections were made in `Envs/clusterEnv.py`:

1. Replaced original-author macOS absolute dataset locations with paths resolved relative to the source file. This lets the checked-in size-30 arrays load on Windows.
2. Removed the `from . import utils` import from this environment after confirming the `Envs/utils.py` file is absent and its only reference in `clusterEnv.py` was commented out.
3. Made `_check_episode_finish()` explicitly return `False` for an incomplete DAG; it previously returned `None`.
4. Fixed the terminal observation's time to match `self.time`. In a full episode, the original code returned a terminal observation at 219.0 while the simulator clock was 234.0. After the fix, both report 234.0.

The final task-completion reward still returns `0` rather than `-time_shift / t_unit`. That omits the final time interval from the RL return and may affect PPO/Actor-Critic learning. It was not changed because this would change the training objective; the issue should be resolved before interpreting new RL results.

## Verified setup and smoke runs

Use the venv described in `INSTALLATION_GUIDE.md`. The following were executed:

1. Direct import/reset/step smoke test: loaded 1,000 DAGs, created a 30-task instance, returned an observation of shape `(98,)`, and accepted a valid action.
2. Full valid-action smoke rollout: seed `random.seed(12345)`, submit the first feasible ready task, otherwise advance simulated time. It completed all 30 tasks in 60 decisions, with makespan 234.0 time units and total task-duration sum 453.0.
3. One episode using the actual SJF helper functions in `baselines_tableau.py`: same Python random seed, 30/30 tasks completed in 60 decisions, makespan 245.0.
4. A compatibility helper was added at `Envs/registration.py` for the `clusterEnv-v0` ID. Gym 0.26.2 registration and a full episode through `gym.make()` passed.
5. AST syntax parsing passed for all 21 Python files.
6. After adding `Envs/registration.py`, the registered `gym.make('clusterEnv-v0').unwrapped` path completed a full 30-task episode in 60 decisions. One SJF-helper episode through this registered environment also completed all 30 tasks. These additional runs selected workflows randomly and are smoke checks, not comparisons.
7. TensorBoard 2.19.0 was installed in the compatibility venv. Importing `PPOagent.py` and running one PPO update on eight transitions passed in a temporary working directory. This is an update-path smoke test, not training.
8. Loaded the existing `PPO_clusterEnv-v0_50MCTS.pth` checkpoint through the `PPOtest.PPO` inference path and completed one 30-task episode in 60 decisions with makespan 198.0. The checkpoint provenance is unknown; this is not a reproduced evaluation.

The one-run makespans are functional smoke observations only. They are not an experimental comparison: there was one workflow, no multiple seeds, no statistical analysis, and the valid-action smoke driver is not a repository baseline.

The exact full SJF smoke command is in `MANUAL_EXECUTION_GUIDE.md`, section “Run a complete workflow episode with the repository's SJF helpers.” It prints:

```text
algorithm=SJF tasks=30 decisions=60 makespan=245.0
```

The smoke command does not create spreadsheet or CSV output. No new experiment results were written under `data/`, and existing repository spreadsheets/figures were not treated as reproduced results.

## Original training/evaluation commands and current blockers

These are the entry points identified in the repository. They are **not currently verified commands**:

```powershell
python DAGs_Generator.py
python PPOagent.py
python PPOtest.py
python baselines_tableau.py
python MonteCarloTreeSearch.py
```

Important reasons not to run them as-is:

- `DAGs_Generator.py`'s current `__main__` requests 100-task arrays, while committed datasets are sizes 10–50 and the generator does not create output directories.
- `clusterEnv-v0` is now registered by `Envs/registration.py` for cluster-based scripts. With Gym 0.26+, its API checker is disabled because this legacy environment does not declare modern Gym spaces/API. The `MyEnv-v0` and `graphEnv-v0` IDs remain unregistered.
- TensorBoard is installed now; `PPOagent.py` emits Chinese console text, so Windows PowerShell may require `$env:PYTHONUTF8 = '1'`.
- Matplotlib remains unavailable. It is not required for the short PPO update.
- The final completion reward is zero, omitting the final time interval from returns. Per the user's decision, preserve the original baseline and postpone full training/evaluation until this issue is addressed.
- Other environment variants still contain original-author absolute macOS paths and missing utility imports.
- Several scripts save fixed output filenames and can overwrite repository artifacts.
- PPO defaults to 300,000 training timesteps, and evaluation defaults to 1,000 episodes. Do not start these long runs as a “smoke test.”

The repository source indicates likely outputs such as PPO checkpoints under `runs/PPO_preTrained/`, PPO evaluation output under `data/makespan_PPO.xls`, baseline spreadsheets under `data/`, and TensorBoard event files under `runs/`. These are expected paths inferred from source, not outputs verified in this phase.

## Recommended next execution steps

1. Create a fully isolated, version-pinned environment rather than sharing global site packages.
2. Decide whether to reproduce the README's legacy stack or maintain the Python 3.12 compatibility experiment; do not combine undocumented versions.
3. Keep the compatibility registration limited to the old cluster environment and validate each cluster-based entry point; resolve registration independently for `MyEnv-v0` and `graphEnv-v0`.
4. Resolve the final-reward accounting issue before training a policy for scientific comparison.
5. Run one training episode/update and one evaluation episode with outputs directed to a temporary experiment folder.
6. Only then run the documented full baseline/evaluation protocol with fixed seeds, preserved raw results, and output paths that do not overwrite existing artifacts.

The above steps are pending, not completed.
