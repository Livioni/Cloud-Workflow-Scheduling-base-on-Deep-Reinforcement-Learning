# Manual Execution Guide (Original Baseline)

This guide records only setup details that have been inspected and the exact blockers observed on the current Windows workstation. It does **not** claim that the original PPO training or full evaluation scripts currently run. No scheduling decision algorithm has been modified.

## 1. What this repository runs

The original project simulates DAG task scheduling on a single abstract CPU/memory resource pool. It is not a real Kubernetes or Edge–Cloud deployment. The main source entry points are `PPOagent.py` (PPO training), `PPOtest.py` (PPO evaluation), and `baselines_tableau.py` (Random/SJF/Tetris-like baselines). A source-level workflow is described in `REPOSITORY_ANALYSIS.md`.

## 2. Verified machine and repository setup

The commands below are PowerShell commands. Run them from the VS Code integrated terminal.

### Open the repository

```powershell
Set-Location 'C:\Users\myahyaoui\Desktop\project_cloud\Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning'
```

This selects the current project directory, which is required because the scripts use repository-relative paths for model and output files.

### Check the available Python

```powershell
py --version
python --version
python -c "import sys; print(sys.executable); print(sys.version)"
```

**Verified result on this machine:** Python 3.12.0 at `C:\Users\myahyaoui\AppData\Local\Programs\Python\Python312\python.exe`. The README instead specifies Python 3.9.7. Python 3.9 is not currently listed as installed.

### Check packages already available

```powershell
python -m pip show torch numpy networkx scipy gym matplotlib tensorboard xlwt xlsxwriter
```

**Verified initial result:** PyTorch 2.11.0+cpu, NumPy 2.4.4, NetworkX 3.6.1, and SciPy 1.17.1 were installed globally. Gym, Matplotlib, TensorBoard, `xlwt`, and `xlsxwriter` were initially missing. No project virtual environment was present at inspection time.

The global interpreter has not been changed. Do not install project packages globally. With approval, a compatibility-test venv was created outside the repository:

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'venvs\cwscheduling-py312'
py -3.12 -m venv --system-site-packages $venv
& "$venv\Scripts\python.exe" -m pip install 'gym==0.26.2' xlwt xlsxwriter
```

These commands were run successfully on this machine. `--system-site-packages` makes the venv see global PyTorch, NumPy, NetworkX, and SciPy; only the additional packages are isolated. This is a compatibility probe, not a fully isolated/reproducible dependency environment. Installed local versions are Gym 0.26.2, `xlwt` 1.3.0, and `xlsxwriter` 3.3.4. The venv shares PyTorch 2.11.0, NumPy 2.4.4, NetworkX 3.6.1, and SciPy 1.17.1 from the global interpreter. Gym prints a warning that it does not support NumPy 2.x. TensorBoard 2.19.0 was later installed and used for a short PPO update; Matplotlib remains absent (the generated sensitivity graphs use SVG and the standard library).

To activate it in a new PowerShell terminal:

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'venvs\cwscheduling-py312'
& "$venv\Scripts\Activate.ps1"
python --version
python -m pip show gym torch numpy networkx scipy xlwt xlsxwriter
```

Activation was **not separately tested**; equivalent commands using the venv's full `python.exe` path were tested. If PowerShell blocks activation, allow scripts only for that terminal process, then activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

This setting ends with the terminal session and does not change the machine-wide execution policy.

The current Torch build is CPU-only (`2.11.0+cpu`). The source can select CUDA if a CUDA-enabled PyTorch build and compatible GPU are available, but a GPU is not required by the simulator itself. Long training speed and hardware requirements have not been benchmarked. At the initial inspection the C: drive had about 37.6 GiB free; after the small venv installs it had about 37.1 GiB free. No external dataset download is required for the inspected size-30 simulation.

Two initial dependency installs were attempted and not completed: Matplotlib's 9.3 MB wheel download timed out, and a separate TensorBoard wheel download stalled and was cancelled. TensorBoard was subsequently installed successfully (2.19.0) for a short PPO update; full PPO training remains unverified. Matplotlib is not required for the current simulator report because plots are generated as SVG.

## 3. Current execution status

The following smoke checks were run:

```powershell
python -c "import torch, numpy, networkx, scipy; print('torch', torch.__version__); print('numpy', numpy.__version__); print('networkx', networkx.__version__); print('scipy', scipy.__version__)"
python -c "import gym"
python -c "import Envs.clusterEnv"
```

The first command succeeded and reported:

```text
torch 2.11.0+cpu
numpy 2.4.4
networkx 3.6.1
scipy 1.17.1
```

The global Gym import failed with:

```text
ModuleNotFoundError: No module named 'gym'
```

The direct environment smoke test succeeded in the venv after narrow fixes to `Envs/clusterEnv.py`: removed the unused import of absent `Envs.utils`, changed its data loaders to repository-relative paths, made `_check_episode_finish()` explicitly return `False` for an incomplete workflow, and fixed the terminal observation time to match the simulator's final clock. No scheduling decision or task execution transition was changed. The final completion still returns reward `0`, so the last time interval is omitted from RL reward accounting; this was left unchanged because it changes training behavior. A compatibility registration helper, `Envs/registration.py`, now registers `clusterEnv-v0` for cluster-based scripts. This was verified with Gym 0.26.2 and does not resolve `MyEnv-v0` or `graphEnv-v0`.

### Re-run the verified direct environment smoke test

From the repository root in PowerShell:

```powershell
$python = Join-Path $env:LOCALAPPDATA 'venvs\cwscheduling-py312\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
@'
from Envs.clusterEnv import clusterEnv

env = clusterEnv()
observation = env.reset()
assert observation.shape == (98,), observation.shape
assert len(env.return_action_list()) > 0
action = env.return_action_list()[0]
next_observation, reward, done, info = env.step(action)
assert next_observation.shape == (98,), next_observation.shape
assert done is False, repr(done)
print(f'loaded_dags={len(env.edges_lib)} tasks={len(env.duration)} observation_shape={observation.shape} action_count={env.return_dim_info()[1]}')
print(f'valid_action={action} reward={reward} done={done} step_returns=4 values')
env.close()
'@ | & $python -
```

**Verified output:** `loaded_dags=1000 tasks=30 observation_shape=(98,) action_count=31`, followed by `valid_action=0 reward=0.0 done=False step_returns=4 values`. Gym also prints its NumPy 2.x compatibility warning. This checks data loading, one reset, and one valid action.

### Run a complete workflow episode with the repository's SJF helpers

This command uses the actual `check_res`, `find_shortest_job`, and `check_ready` helper functions defined in `baselines_tableau.py`. It runs one episode only; the fixed Python random seed makes the dataset selection repeatable for this smoke check but is not a multi-run scientific evaluation.

```powershell
$python = Join-Path $env:LOCALAPPDATA 'venvs\cwscheduling-py312\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
@'
import random
from Envs.clusterEnv import clusterEnv
import baselines_tableau as baselines

random.seed(12345)
env = clusterEnv()
state = env.reset()
steps = 0
while steps < 10000:
    if baselines.check_res(state):
        candidate = baselines.find_shortest_job(state)
        action = candidate if baselines.check_ready(state, candidate) else -1
    else:
        action = -1
    state, reward, done, info = env.step(action)
    if info[0] is False:
        raise RuntimeError(f'SJF produced invalid action {action} at step {steps}')
    steps += 1
    if done:
        break
else:
    raise RuntimeError('SJF episode exceeded 10,000 actions')

assert done is True
assert len(env.done_job) == 30
assert float(state[0]) == env.time
print(f'algorithm=SJF tasks={len(env.done_job)} decisions={steps} makespan={env.time}')
env.close()
'@ | & $python -
```

**Verified output:** `algorithm=SJF tasks=30 decisions=60 makespan=245.0`. This is a functional smoke result for one workflow, not evidence that SJF outperforms another policy.

For comparison, a simple valid-action-only smoke policy (submit the first feasible ready task, otherwise advance simulated time) completed the same seeded workflow in 60 decisions with makespan `234.0`. This policy is only a test driver and is not one of the repository's named baselines.

**Terminal state correction:** Before the fix, the final `step()` returned a state timestamp of `219.0` while the simulator clock was `234.0`; after correcting the terminal observation, the same workflow returned `234.0` in both places. The terminal reward remains zero and should be considered when interpreting any future RL training.

### Evaluate one episode from an existing PPO checkpoint

The repository includes a checkpoint named `PPO_clusterEnv-v0_50MCTS.pth`. A one-episode inference smoke test loaded it through the existing `PPOtest.PPO` implementation, completed all 30 tasks in 60 decisions, and observed makespan `198.0`. This is a single episode using an existing checkpoint of unverified provenance; it is not a reproduced or statistically meaningful evaluation.

## 4. Original execution procedure documented by the repository

The README describes this conceptual order:

1. Register a custom environment with Gym. The original source did not include this registration; a compatibility helper now handles `clusterEnv-v0` only.
2. Run the DAG generator to create training/testing datasets.
3. Change the environment's dataset paths.
4. Train with PPO/DRL.
5. Evaluate a trained policy.
6. Run baseline and MCTS experiments.

The checked-out files differ from some README path references. The corresponding root-level source files are `DAGs_Generator.py`, `PPOagent.py`, `PPOtest.py`, `baselines_tableau.py`, and `MonteCarloTreeSearch.py`.

The commands below are **not verified and should not be run yet**:

```powershell
python DAGs_Generator.py
python PPOagent.py
python PPOtest.py
python baselines_tableau.py
python MonteCarloTreeSearch.py
```

Reasons: the current generator's `__main__` requests 100-task data while committed generated datasets cover 10–50 task DAGs; output directories are not created by the generator; package compatibility is not established; environment registration and imports in some variants are incomplete; other environment classes still use macOS paths; and some scripts save fixed filenames that could overwrite existing files.

**Expected output files based on source only, not verified execution:** PPO checkpoints under `runs/PPO_preTrained/`, PPO evaluation output under `data/makespan_PPO.xls`, baseline spreadsheets under `data/`, and training event logs under `runs/`. Existing files with similar names are repository artifacts and are not proof that these commands ran successfully on this workstation.

## 5. Dataset inventory

The `.npy` arrays are already present; no external dataset was identified as a prerequisite. For train and test collections at task sizes 10, 20, 30, 40, and 50, there are 1,000 samples in each of the edge, duration, and demand arrays. An additional `npy/test_datasheet/test/` collection contains 100 samples with 30 tasks each.

The `.npy` files were inspected with NumPy and their dimensions were confirmed. Edge arrays use `dtype=object`; the environment source loads them with `allow_pickle=True`. Do not regenerate or overwrite these files as part of initial reproduction.

## 6. Next setup decision and pending steps

The documented baseline runtime is Python 3.9.7 with PyTorch 1.10, Gym 0.21.0, and NetworkX 2.6.3. The user selected a Python 3.12 compatibility attempt rather than installing another system Python. The direct environment smoke test passed, but the venv shares global packages and Gym 0.26.2 warns about NumPy 2.x. It is not yet a reproducible baseline environment.

The verification sequence still needed is:

1. Establish a fully isolated and version-pinned environment rather than one sharing globally installed packages.
2. The cluster registration, TensorBoard install, short PPO update, and single-checkpoint inference smoke checks are complete. Resolve other custom environment IDs separately if testing those variants.
3. Preserve the baseline terminal reward unchanged for now. Full training is postponed until the reward-accounting issue is explicitly revisited; no reward change has been approved.
4. Record exact package versions, console output, generated files, and any further compatibility fixes before expanding experiments.

The environment reset/step, registered Gym path, single SJF workflow, eight-transition PPO update, and one checkpoint inference episode are complete smoke checks. Full PPO training, PPO's 1,000-episode evaluation, and the original 1,000-episode baseline suite remain unverified and are postponed until the terminal reward accounting decision is revisited. Single-episode metrics are not reproducible experimental conclusions.

## 7. Troubleshooting facts confirmed so far

| Symptom | Confirmed cause | Current status |
|---|---|---|
| `No module named 'gym'` in the global interpreter | Gym was initially absent from system Python. | Confirmed; Gym 0.26.2 is now installed only in the compatibility venv. |
| Dataset path failure in `clusterEnv` on Windows | Original code used macOS absolute paths. | Fixed for `clusterEnv` with source-relative paths; other environments remain unchanged. |
| Import failure for `utils` in `clusterEnv` | `Envs/utils.py` is absent and the import was unused in active code. | Fixed by removing the unused import in `clusterEnv`; other variants still require review. |
| `gym.make(...)` cannot find a custom environment | Original source had no registration. | `clusterEnv-v0` is now registered in project source and a full episode through `gym.make()` passed. `MyEnv-v0` and `graphEnv-v0` remain unresolved. |
| PPO import/update fails in Windows PowerShell | TensorBoard may be missing, or Chinese output may not be encodable by cp1252. | TensorBoard 2.19.0 is installed. `PYTHONUTF8=1` allowed the eight-transition PPO update smoke check. |
| Legacy API/version conflicts | README pins Python 3.9.7/Gym 0.21, while the compatibility venv uses Python 3.12/Gym 0.26.2 and global NumPy 2.4.4. | Direct class smoke test passed; full wrapper/training compatibility is unverified and Gym emits a compatibility warning. |
| Terminal observation time was behind the simulator clock | `clusterEnv.step()` returned the previous state when the last task completed. | Fixed by updating the terminal state's time; verified with a full episode. Final completion reward remains zero. |

This guide will be updated only after commands are executed and their results are observed.

## 8. Run the separate Edge–Cloud simulator

The simulator is independent from the original Gym environment. It uses only repository workload arrays and explicit simulation parameters; it does not create Kubernetes pods or contact a cluster. Run the tests and a small held-out workload sample from the repository root:

```powershell
$python = Join-Path $env:LOCALAPPDATA 'venvs\cwscheduling-py312\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $python -m unittest tests.test_edge_cloud_simulator -v
& $python edge_cloud_experiment.py --config configs\edge_cloud_dataset.json --dataset-size 50 --dataset-split test --dataset-indices 0 1 2 3 4 5 6 7 8 9 --data-size-per-edge 1 --observation-delay 5 --output results\stale_test.csv --seed 100
```

The command evaluates FCFS, SJF, minimum-finish, and stale minimum-finish policies on the same selected DAGs. `--data-size-per-edge 1` is an invented uniform payload assumption because the committed DAG arrays contain no edge payload volumes; `--observation-delay 5` is likewise a sensitivity parameter, not measured telemetry. Raw and summary CSV files are written under `results\`. Seventeen unit tests passed for the current simulator revision; full PPO learning and live Kubernetes integration remain unverified and are not part of this command.

### Full sensitivity grid and report

```powershell
$python = Join-Path $env:LOCALAPPDATA 'venvs\cwscheduling-py312\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$indices = 0..999
& $python edge_cloud_experiment.py --config configs\edge_cloud_dataset.json --dataset-size 50 --dataset-split test --dataset-indices $indices --replicates 1 --seed 2300 --bootstrap-samples 500 --output results\sensitivity_test50_final.csv --sensitivity-raw-output results\sensitivity_test50_raw.csv --log-output results\sensitivity_run.log --sensitivity-data-sizes 1 10 --sensitivity-observation-delays 0 5 --sensitivity-link-scales 0.5 2 --sensitivity-node-capacity-scales 0.8 1.2
& $python generate_sensitivity_report.py --summary results\sensitivity_test50_final.csv --raw results\sensitivity_test50_raw.csv --report SENSITIVITY_ANALYSIS_REPORT.md --figure-dir results
```

The grid evaluates 16 parameter combinations against all 1,000 selected DAG indices and preserves per-policy/per-DAG metrics. `SENSITIVITY_ANALYSIS_REPORT.md` and the SVG figures must be generated after the CSV run completes. These are assumption-sensitivity experiments, not measurements from a real cluster.
