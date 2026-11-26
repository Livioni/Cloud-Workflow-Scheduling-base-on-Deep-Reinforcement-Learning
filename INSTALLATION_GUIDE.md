# Installation Guide

## Status and scope

This guide documents a Windows/Python 3.12 compatibility probe, not the repository author's original pinned runtime. The README requests Python 3.9.7, PyTorch 1.10, Gym 0.21.0, and NetworkX 2.6.3. The workstation had Python 3.12.0 and PyTorch 2.11.0+cpu, NumPy 2.4.4, NetworkX 3.6.1, and SciPy 1.17.1. The original Python 3.9 runtime was not installed.

The user approved trying the Python 3.12 path instead of installing another system Python. The venv below was created outside the repository so it does not add environment files to source control. It uses `--system-site-packages` to reuse the existing ML/numerical packages; this is convenient for a compatibility test but is not fully isolated or sufficient for a reproducibility claim.

## Prerequisites

- Windows 10/11 and PowerShell (tested on Windows).
- Visual Studio Code is optional; the project can be run from any PowerShell terminal.
- Python 3.12.0 was available and used in the test. The README's Python 3.9.7 was not installed.
- The repository's `.npy` DAG datasets are included; no external dataset download was needed for the size-30 smoke runs.
- The tested PyTorch build is CPU-only. The simulator does not require a GPU; long training duration has not been measured.
- At initial inspection, the C: drive had about 37.6 GiB free; after the test venv installs, about 37.1 GiB remained.
- No environment variables or credentials are required for the direct simulation.
- No Kubernetes installation is required: the repository does not connect to Kubernetes.

## Create the compatibility virtual environment

Open PowerShell and select the repository:

```powershell
Set-Location 'C:\Users\myahyaoui\Desktop\project_cloud\Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning'
```

Create the venv and install the small runtime set used for direct simulation and baseline-helper imports:

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'venvs\cwscheduling-py312'
py -3.12 -m venv --system-site-packages $venv
& "$venv\Scripts\python.exe" -m pip install 'gym==0.26.2' xlwt xlsxwriter
& "$venv\Scripts\python.exe" -m pip install 'tensorboard==2.19.0'
```

These commands were run successfully. Gym 0.26.2, `xlwt` 1.3.0, and `xlsxwriter` 3.3.4 are installed in the venv. Because `--system-site-packages` is set, the following are visible from the global interpreter rather than independently pinned in the venv: PyTorch 2.11.0+cpu, NumPy 2.4.4, NetworkX 3.6.1, and SciPy 1.17.1.
TensorBoard 2.19.0 and its dependencies were subsequently installed in the venv. TensorBoard import and `pip check` both pass.

Activate the venv in a new PowerShell terminal:

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'venvs\cwscheduling-py312'
& "$venv\Scripts\Activate.ps1"
python --version
```

The activation commands were not separately tested; all test commands used the full path to `python.exe`. If activation is blocked by PowerShell, permit scripts only for the current process and retry:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
& "$venv\Scripts\Activate.ps1"
```

The process-scoped policy ends when that terminal closes; it does not change the machine-wide policy.

## Verify the environment

Run:

```powershell
$python = Join-Path $env:LOCALAPPDATA 'venvs\cwscheduling-py312\Scripts\python.exe'
& $python --version
& $python -m pip show gym torch numpy networkx scipy xlwt xlsxwriter
& $python -m pip check
```

Verified package versions are Gym 0.26.2, PyTorch 2.11.0+cpu, NumPy 2.4.4, NetworkX 3.6.1, SciPy 1.17.1, `xlwt` 1.3.0, `xlsxwriter` 3.3.4, and TensorBoard 2.19.0. `pip check` reported `No broken requirements found.` Gym still prints a warning that it is unmaintained and does not support NumPy 2.x; a clean `pip check` does not prove runtime compatibility.

## Dependency status

Matplotlib remains unavailable in the compatibility venv:

- A broad install using `python -m pip install --timeout 180 --retries 5 'gym==0.26.2' matplotlib tensorboard xlwt xlsxwriter` timed out while downloading the Matplotlib wheel.
- An earlier separate TensorBoard download stalled and was cancelled. A later pinned install of `tensorboard==2.19.0` completed successfully.

Matplotlib is not needed to load the committed dataset, run the direct simulation, or execute the tested PPO policy update. TensorBoard is imported by `PPOagent.py`; its installation unblocked module import and a short PPO update smoke check.

## Known compatibility limitations

1. The requested source-compatible Python 3.9.7/Gym 0.21.0/PyTorch 1.10 setup has not been recreated.
2. The compatibility venv uses current globally installed NumPy 2.4.4 with Gym 0.26.2; Gym warns about this combination.
3. The environment code uses the legacy four-return Gym `step()` API. `Envs/registration.py` disables Gym 0.26's checker for `clusterEnv-v0`; the registered environment and a short PPO policy update were smoke-tested, but full training/evaluation was not.
4. Windows' default cp1252 output encoding cannot print some Chinese messages in `PPOagent.py`. Set `$env:PYTHONUTF8 = '1'` in PowerShell before running the script.
5. Custom IDs for `MyEnv-v0` and `graphEnv-v0` are still not registered; only `clusterEnv-v0` has the compatibility helper.
6. This project has no dependency manifest or lockfile. The versions above describe the observed compatibility probe only, not a fully reproducible environment.

Do not treat this compatibility environment as the final scientific execution setup until the legacy runtime, Gym registration/API, and full dependency set are resolved and documented.
