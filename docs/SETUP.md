# Environment setup

How to build, verify and update the development environment, and how to attach an ext4
data drive under WSL2.

## Requirements

- Linux on x86_64 (native or WSL2) with glibc 2.31 or newer; the CARLA 0.9.16 client wheel
  is `manylinux_2_31`.
- An NVIDIA GPU with a driver recent enough for CUDA 12.4. The PyTorch wheels bundle their
  own CUDA runtime.
- conda (Miniconda or Miniforge).
- A CARLA 0.9.16 server. Under WSL2 it can run on the Windows host, reached at the WSL
  gateway address on port 2000.

## Build

```bash
conda env create -f environment.yml          # creates the env "rwkv"
conda activate rwkv
pip install -e . --no-deps                   # the project itself, editable
pre-commit install
make env-check                               # must end with "OK: runs/smoke/..."
```

Use `conda env create -f environment.yml -n <name>` to build a second copy next to an
existing one.

## What each dependency file does

| File | Role | Edited by |
| --- | --- | --- |
| `pyproject.toml` | Declared dependencies with version ranges. Extras: `dev`, `simulator` | Hand |
| `requirements-lock.txt` | Exact versions with sha256 hashes for every package | `make lock` only |
| `environment.yml` | conda supplies Python 3.11 and pip; pip installs the lock | Hand (rarely) |
| `environment.lock.yml` | Exact record of the built conda environment | `conda env export` |

conda installs only the interpreter, so one installer (pip) owns every Python package.

## Verify

`make env-check` runs, in order:

1. `scripts/verify_env.py`: imports, CUDA, a forward pass; exits non-zero on failure.
2. The offline test suite (`pytest -m "not online"`).
3. A one-epoch GPU smoke training run on `data/toy` with `configs/smoke_toy.yaml`.
4. `scripts/check_smoke_run.py`: finite losses, non-zero gradients, a validation pass and
   a checkpoint.

## Change dependencies

1. Edit `pyproject.toml`.
2. `make lock` (needs [uv](https://docs.astral.sh/uv/)). Existing pins are kept unless
   the change requires new versions; re-running on an unchanged `pyproject.toml` gives a
   byte-identical file.
3. Rebuild a test copy (`conda env create -f environment.yml -n rwkv-test`), run
   `make env-check` in it, then commit `pyproject.toml` and `requirements-lock.txt`
   together.

## WSL memory limits

By default WSL2 can take most of the host's memory, leaving too little for Windows and a
CARLA server running there. Cap it in `%UserProfile%\.wslconfig`, leaving roughly 12 GB
for Windows and the simulator:

```ini
[wsl2]
vmIdleTimeout=-1
memory=<host RAM minus about 12 GB>
swap=8GB

[experimental]
autoMemoryReclaim=gradual
```

Changes take effect after `wsl --shutdown` from PowerShell, which also closes every WSL
session and unmounts any attached data drive. The WSL system disk grows as files are
written inside WSL and does not shrink on its own, so keep datasets and caches on a
separate data drive.

## Data drive under WSL

Windows cannot read ext4, so an ext4 data drive is passed through to WSL. Find its disk
number `<N>` with `Get-Disk`, then attach it from an Administrator PowerShell:

```powershell
Set-Disk -Number <N> -IsOffline $true
wsl --mount \\.\PHYSICALDRIVE<N> --partition 1 --type ext4
```

It appears at `/mnt/wsl/PHYSICALDRIVE<N>p1`. Add `--options "ro"` for read-only analysis.
Use exactly one backslash between the dot and `PHYSICALDRIVE`. If Windows has assigned the
partition a drive letter, `wsl --mount` fails with `E_FAIL` until the disk is offline.

Detach and give the disk back to Windows:

```powershell
wsl --unmount \\.\PHYSICALDRIVE<N>
Set-Disk -Number <N> -IsOffline $false
```

## Second machine (Apple silicon)

An Apple silicon Mac can run tests, data audits, offline evaluation and small baselines
while the GPU machine runs CARLA. It cannot run CARLA (no macOS build) or the Triton
kernels; PyTorch uses the Apple GPU through MPS.

### Code

The Mac's clone accepts pushes straight from the GPU machine and updates its working
tree. With SSH access to the Mac under the host alias `mac`:

```bash
# on the Mac, once
git init ~/projects/neuromorphic-vision-rwkv
git -C ~/projects/neuromorphic-vision-rwkv config receive.denyCurrentBranch updateInstead
# on the GPU machine
git remote add mac mac:projects/neuromorphic-vision-rwkv
git push mac <branch>          # then, the first time only: git checkout <branch> on the Mac
```

Keep the Mac's working tree clean; a push that would overwrite local changes is refused.

### Environment

```bash
conda env create -f environment-macos.yml        # env name: rwkv
conda run -n rwkv pip install -e . --no-deps
conda run -n rwkv make toy-data
conda run -n rwkv make env-check                 # trains on "mps (Apple GPU)"
```

`requirements-lock-macos.txt` pins the same versions as `requirements-lock.txt`, minus
CUDA and NVIDIA libraries, Triton, the CARLA client and the unused `rwkv` package.
Re-pin with `make lock-macos` after `make lock`.

### Large data

Move datasets on an external drive rather than over the network. macOS needs a
third-party driver to read ext4, and such drivers are often read-only, so write data on
the Linux machine.

### Numerical agreement between machines

The same seed gives nearly, but not bit-for-bit, identical runs on the two CPU
architectures: freshly initialised weights already differ by a few float32 units in the
last place (about 6e-9), and a seeded forward pass differs by up to 4e-7. Compare runs
across machines with tolerances, not exact equality.
