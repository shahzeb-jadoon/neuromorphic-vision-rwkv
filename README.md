# Neuromorphic Vision-RWKV

End-to-end driving in the CARLA simulator with a Vision-RWKV policy. The question is
whether RWKV's linear-time recurrence, designed for language modelling, can replace
quadratic-cost vision transformers for real-time driving perception, with a path toward
efficient edge and neuromorphic hardware.

**Status:** research prototype, being rebuilt. The first trained model (April 2026, see
[docs/RUNS.md](docs/RUNS.md)) did not learn to steer: its steering error matches that of
always predicting zero. The labels, the model and the training loop are being fixed before
the next training run.

## Layout

| Path | Contents |
| --- | --- |
| `src/data/` | Expert data collection, HDF5 episode schema, clip dataset, route generation |
| `src/models/` | Vision-RWKV driver: CNN tokenizer, temporal cores, navigation fusion, policy head |
| `src/training/` | Trainer with checkpointing and resume |
| `agents/expert/` | Privileged expert driver that records training data |
| `agents/learned/` | Agent that drives with a trained model |
| `agents/navigation/` | Route planner vendored from CARLA 0.9.16 (MIT) |
| `scripts/` | Collection, training, evaluation, analysis and environment checks |
| `configs/` | Collection and training configurations |
| `tests/` | Offline tests; tests marked `online` need a CARLA server |
| `docs/` | Setup, run registry, design notes and literature analyses |

## Quickstart

```bash
conda env create -f environment.yml
conda activate rwkv
pip install -e . --no-deps
make env-check
```

`make env-check` verifies the environment, runs the offline tests and trains for one epoch
on a small synthetic dataset. [docs/SETUP.md](docs/SETUP.md) covers requirements,
dependency updates, data drives under WSL and an Apple silicon second machine. Collecting
data and closed-loop evaluation need a CARLA 0.9.16 server; `make help` lists the other
targets.

## Licence

Code: [Apache License 2.0](LICENSE). Datasets and model weights:
[CC BY-NC 4.0](DATA_LICENSE). The route planner in `agents/navigation/` is vendored from
CARLA under the MIT licence.
