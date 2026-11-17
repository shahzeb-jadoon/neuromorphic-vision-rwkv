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
| Data | Schema v3, 1,099 episodes, 3,624,673 frames |
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

The files live on the machine that produced them, with a second copy on a separate drive
whose sha256 sums matched the table above on 2026-11-17.

### Offline evaluation: v3 baselines of record (2026-11-17)

`best.pt` on the run's validation split, recorded before the v3 dataset was retired:
219 episodes, 1,457 clips of 32 frames (stride 512, resized to 160×960), scored on
frames 8–31 after the burn-in, 34,968 frames in all. Baselines are fitted on the
training split's labels (880 episodes, 2,911,535 frames). Turning frames have
|expert steering| > 0.1.

```bash
python scripts/evaluate_offline.py --checkpoint runs_fast/fastsafe_20260424_093210/checkpoints/best.pt
```

The trainer's own validation metric is reproduced exactly: steering MAE 0.054577 and
acceleration MAE 0.201928 over the first 50 batches, frame-weighted, as logged at
step 6,000.

Pooled, with the training frame weights (stationary frames count 0.1; turning frames
are 18.2% of the weight):

| Predictor | Steering MAE | Turn steering MAE | Turn sign accuracy | Accel MAE | Steering corr. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best.pt` | 0.0576 | 0.2375 | 0.506 | 0.2000 | 0.026 |
| Always zero | 0.0547 | 0.2371 | 0.000 | 0.4303 | — |
| Always the training mean | 0.0565 | 0.2376 | 0.453 | 0.3934 | — |
| Per-command training mean | 0.0594 | 0.2165 | 0.681 | 0.3688 | 0.241 |

Pooled, every frame equal (turning frames are 15.9% of frames):

| Predictor | Steering MAE | Turn steering MAE | Turn sign accuracy | Accel MAE | Steering corr. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best.pt` | 0.0526 | 0.2456 | 0.525 | 0.3686 | 0.026 |
| Always zero | 0.0500 | 0.2455 | 0.000 | 0.4544 | — |
| Always the training mean | 0.0518 | 0.2456 | 0.496 | 0.4414 | — |
| Per-command training mean | 0.0596 | 0.2261 | 0.677 | 0.4058 | 0.241 |

What the numbers say about `best.pt`:

- Steering is no better than always predicting zero, and on turning frames its sign is
  right about half the time. A blind per-command average turns better.
- It ignores the navigation command: mean steering on LEFT is +0.004 (expert −0.057)
  and on RIGHT +0.005 (expert +0.065), weighted.
- It ignores the image layout: the camera-swap score is −0.995, so mirroring the
  panorama leaves the steering unchanged. Its steering spread is 0.009 against the
  expert's 0.122.
- It did learn something about speed: acceleration MAE 0.2000, against 0.3688 for the
  best baseline.

The JSON report and the per-frame arrays are in the run's `eval/` folder.

### Offline evaluation on the v3 testbed cache (2026-11-18)

The full v3 recording has been retired; the testbed cache described in
[DATA_V3.md](DATA_V3.md) replaces it. The cache keeps every signal at 20 Hz, so the tables
above still hold exactly: rebuilt from the cache, the 34,968 scored frames (steering,
acceleration, command and weight) and the 2,911,535 training frames are identical to the
record, and every baseline score matches it to the last digit.

The cache's images are at 10 Hz, so the evaluator run on it scores a different sample:

```bash
python scripts/evaluate_offline.py --checkpoint runs_fast/fastsafe_20260424_093210/checkpoints/best.pt \
    --data-dir <testbed cache> --out runs_fast/fastsafe_20260424_093210/eval/best_testbed_offline.json
```

Same split (219 validation and 880 training episodes) and clip settings, but a 32-frame
clip now spans 3.2 s, giving 749 clips and 17,976 scored frames. Turning frames are a
smaller share (15.0% of the weight, 13.3% of frames), so every MAE is lower than above.
These numbers are the reference for models trained and evaluated on the cache. The
trainer's metric over the first 50 batches is steering MAE 0.0511 and acceleration MAE
0.1801.

Pooled, with the training frame weights:

| Predictor | Steering MAE | Turn steering MAE | Turn sign accuracy | Accel MAE | Steering corr. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best.pt` | 0.0508 | 0.2311 | 0.471 | 0.1760 | 0.036 |
| Always zero | 0.0474 | 0.2308 | 0.000 | 0.3847 | — |
| Always the training mean | 0.0493 | 0.2313 | 0.450 | 0.3479 | — |
| Per-command training mean | 0.0525 | 0.2119 | 0.677 | 0.3231 | 0.229 |

Pooled, every frame equal:

| Predictor | Steering MAE | Turn steering MAE | Turn sign accuracy | Accel MAE | Steering corr. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best.pt` | 0.0479 | 0.2431 | 0.501 | 0.3196 | 0.036 |
| Always zero | 0.0449 | 0.2432 | 0.000 | 0.4141 | — |
| Always the training mean | 0.0469 | 0.2433 | 0.487 | 0.3973 | — |
| Per-command training mean | 0.0543 | 0.2249 | 0.673 | 0.3610 | 0.229 |

Every finding above holds on the cache: steering no better than always zero, the turn sign
right about half the time, mean steering +0.004 on LEFT and +0.005 on RIGHT (expert −0.050
and +0.050, weighted), camera-swap score −0.997, and acceleration MAE 0.1760 against 0.3231
for the best baseline.

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
