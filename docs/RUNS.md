# Run registry

Provenance for every training run whose results are reported anywhere. Run directories and
checkpoints are git-ignored and live only on the machine that produced them, so this file
is the durable record of what they contained.

Each entry records the code commit, data, configuration, stopping condition and the
sha256 of every artifact. Later runs append new entries in the same format.

---

## R-2026-04-A — `fastsafe_20260424_093210` (reference run)

| Field | Value |
| --- | --- |
| Directory | `runs_fast/fastsafe_20260424_093210/` (local only, git-ignored) |
| Code | `f725d42` (`0130497` before a November 2026 history rewrite) — most recent commit when the run started; working-tree state was not captured |
| Code evidence | Run started 2026-04-24 09:32 local time (UTC−4, the `-0400` offset on that day's commits), after `f725d42` at 07:17 `-0400`. The logged speed loss at step 20 (0.126) matches the km/h → [0, 1] target normalisation introduced by `f725d42`. |
| Data | `/mnt/data/carla_data`, schema v3, 1,099 episodes, 3,624,673 frames |
| Split | episode-level, seed 42, validation fraction 0.2 (219 validation episodes) |
| Sampling | clips of 32 frames, stride 512, resized to 160×960, weighted sampler, flip augmentation |
| Model | `VisionRWKVDriver`, 13.43M parameters, 6 encoder layers, width 384, patch size 16 |
| Optimisation | AdamW, lr 1e-4, weight decay 1e-2, batch 8, cosine schedule with 1,000 warm-up steps, fp16 autocast with GradScaler, `torch.compile` on the encoder |
| Stop | manual SIGINT (signal 2) at step 10,217, 2026-04-24 21:25 local time |
| Resume | `config_resolved.yaml` lists `resume: …/checkpoints/latest.pt`, which indicates at least one relaunch with auto-resume. `events.csv` has no `resume` row, so the resume step cannot be confirmed from the logs. |
| Selected checkpoint | `best.pt`: step 6,000, epoch 8, validation total loss 1.221815 |
| Post-run findings | Gradient norm is exactly 0 from step 7,780 to 10,217: the fp16 loss scale fell from 1024 (step 6,000) to 2^-75 (step 10,217) through repeated non-finite steps, so weights stopped updating after step ~7,780. Re-evaluated on the validation split, `best.pt` has steering MAE 0.0576 (pooled) against 0.0547 for a constant zero prediction. |

### Artifacts

| File | Bytes | Step | Epoch | Loss scale | sha256 |
| --- | ---: | ---: | ---: | --- | --- |
| `checkpoints/best.pt` | 161,325,486 | 6,000 | 8 | 1024 | `48832b1a3ade2f2e9723795f5e075acaf10789b2dfe9c70231437bb6427723f4` |
| `checkpoints/latest.pt` | 161,328,564 | 10,217 | 13 | 2^-75 | `fcd72c7eeeb380d66ed7e19110b2e21b0c99d511f877ab915a7a677fc84f5433` |
| `checkpoints/step_00009500_epoch_0012_step.pt` | 161,326,076 | 9,500 | 12 | 2^-52 | `1568622a2dbadc68ee24a215c07c2e4c755c272e3799ac37e2c3b61aa8556e82` |
| `checkpoints/step_00009646_epoch_0012_epoch.pt` | 161,326,666 | 9,646 | 12 | 2^-57 | `c51d292eb12c069635f70ee5b4e669aee18c50938505604f616f5a3009f8bf8d` |
| `checkpoints/step_00010000_epoch_0013_step.pt` | 161,326,140 | 10,000 | 13 | 2^-71 | `183e745b446f7f19a21226339292f9241dc2f06aa4e3e14f118eb512f5b0574f` |
| `checkpoints/step_00010000_epoch_0013_val.pt` | 161,325,614 | 10,000 | 13 | 2^-71 | `1ecc9504e98a90b82307065bb12e5d3032cbb943c54bfe6de5ecdd5e7d1ee10d` |
| `checkpoints/step_00010217_epoch_0013_signal_2.pt` | 161,328,564 | 10,217 | 13 | 2^-75 | `fcd72c7eeeb380d66ed7e19110b2e21b0c99d511f877ab915a7a677fc84f5433` |
| `config_resolved.yaml` | 1,980 | — | — | — | `d32effe319b37c76973ac0c583101b1087341cd06336556b3bbaaf31cffd5ad3` |
| `logs/metrics.csv` | 111,500 | — | — | — | `065d1ea83942cc65637196b837b4e95ac1af809ffa413c60b70452179d37aac2` |
| `logs/events.csv` | 3,713 | — | — | — | `369cb3e1579120f01de5b6860492526065c8b815ea4f9e78e75f51b7ec435724` |

`latest.pt` and `step_00010217_epoch_0013_signal_2.pt` are byte-identical.

### Verify

```bash
cd runs_fast/fastsafe_20260424_093210 && sha256sum -c - <<'EOF'
48832b1a3ade2f2e9723795f5e075acaf10789b2dfe9c70231437bb6427723f4  checkpoints/best.pt
fcd72c7eeeb380d66ed7e19110b2e21b0c99d511f877ab915a7a677fc84f5433  checkpoints/latest.pt
1568622a2dbadc68ee24a215c07c2e4c755c272e3799ac37e2c3b61aa8556e82  checkpoints/step_00009500_epoch_0012_step.pt
c51d292eb12c069635f70ee5b4e669aee18c50938505604f616f5a3009f8bf8d  checkpoints/step_00009646_epoch_0012_epoch.pt
183e745b446f7f19a21226339292f9241dc2f06aa4e3e14f118eb512f5b0574f  checkpoints/step_00010000_epoch_0013_step.pt
1ecc9504e98a90b82307065bb12e5d3032cbb943c54bfe6de5ecdd5e7d1ee10d  checkpoints/step_00010000_epoch_0013_val.pt
fcd72c7eeeb380d66ed7e19110b2e21b0c99d511f877ab915a7a677fc84f5433  checkpoints/step_00010217_epoch_0013_signal_2.pt
d32effe319b37c76973ac0c583101b1087341cd06336556b3bbaaf31cffd5ad3  config_resolved.yaml
065d1ea83942cc65637196b837b4e95ac1af809ffa413c60b70452179d37aac2  logs/metrics.csv
369cb3e1579120f01de5b6860492526065c8b815ea4f9e78e75f51b7ec435724  logs/events.csv
EOF
```

### Storage

These files exist only on the machine that produced them. They have no second copy.

---

## Other local runs (not reported)

Short smoke tests and aborted attempts from 22–24 April 2026. None of their numbers are
reported.

| Directory | Checkpoints | Last logged step |
| --- | ---: | --- |
| `runs/vrwkv_e12d768_s32b4_lr1e-4_wd1e-2_res320x1920_seed42_20260422_052743` | 0 | 500 |
| `runs_fast/fastsafe_20260422_130852` | 2 | 140 |
| `runs_fast/fastsafe_20260422_132106` | 3 | none |
| `runs_fast/fastsafe_20260422_132441` | 7 | 6,000 |
| `runs_fast/fastsafe_20260424_051334` | 2 | 60 |
| `runs_fast/fastsafe_20260424_060752` | 3 | none |
| `runs_fast/fastsafe_20260424_061041` | 2 | 100 |
| `runs_fast/fastsafe_20260424_061914` | 3 | none |
| `runs_fast/fastsafe_20260424_062525` | 4 | 820 |
| `runs_fast/fastsafe_20260424_081138` | 0 | 20 |
| `runs_fast/fastsafe_20260424_083958` | 0 | none |
| `runs_fast/fastsafe_20260424_092102` | 0 | none |
