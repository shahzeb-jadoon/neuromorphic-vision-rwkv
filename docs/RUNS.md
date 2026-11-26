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
| Post-run findings | Gradient norm is exactly 0 from step 7,780 to 10,217: the fp16 loss scale fell from 1024 (step 6,000) to 2^-75 (step 10,217) through repeated non-finite steps, so weights stopped updating after step ~7,780. Re-evaluated on the validation split, `best.pt` has steering MAE 0.0576 (pooled) against 0.0547 for a constant zero prediction. Clips started only every 512 frames, so the run could train on about 6% of the training frames ([LABELS.md](LABELS.md#clip-sampling)). |

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

### Offline evaluation with the Phase 1 labels (2026-11-18)

Phase 1 changed what is scored ([LABELS.md](LABELS.md)): the 3 s before a collision and
the final stop of a timed-out episode are no longer used, red-light stops in the wrong place
weigh 0, and commands are derived from the path and the junction outlines. Same split and
clip settings as above, now 711 clips and 17,064 scored frames. These numbers are the
reference for Phase 2. The trainer's metric over the first 50 batches is steering MAE 0.0522
and acceleration MAE 0.1966.

```bash
python scripts/evaluate_offline.py --checkpoint runs_fast/fastsafe_20260424_093210/checkpoints/best.pt \
    --out runs_fast/fastsafe_20260424_093210/eval/best_testbed_phase1.json
```

Pooled, with the training frame weights:

| Predictor | Steering MAE | Turn steering MAE | Turn sign accuracy | Accel MAE | Steering corr. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best.pt` | 0.0516 | 0.2295 | 0.491 | 0.1990 | 0.029 |
| Always zero | 0.0481 | 0.2293 | 0.000 | 0.3860 | — |
| Always the training mean | 0.0502 | 0.2299 | 0.449 | 0.3200 | — |
| Per-command training mean | 0.0531 | 0.2037 | 0.702 | 0.2870 | 0.264 |

Pooled, every frame equal:

| Predictor | Steering MAE | Turn steering MAE | Turn sign accuracy | Accel MAE | Steering corr. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best.pt` | 0.0480 | 0.2388 | 0.513 | 0.3281 | 0.029 |
| Always zero | 0.0450 | 0.2392 | 0.000 | 0.4103 | — |
| Always the training mean | 0.0471 | 0.2394 | 0.482 | 0.3760 | — |
| Per-command training mean | 0.0548 | 0.2133 | 0.712 | 0.3297 | 0.264 |

The junction commands carry more steering information than the recorded ones: the
per-command baseline gets the turn sign right on 70.2% of weighted turning frames (67.7% with
the recorded commands), and its steering correlates 0.264 with the expert's (0.229).
`best.pt` still steers no better than always zero (camera-swap score −0.998). Its
acceleration MAE rises from 0.1760 to 0.1990: it was trained on the recorded commands, under
which STRAIGHT was 2% of frames rather than 23%.

---

## Training numerics since Phase 2

The April run's loss scale collapsed under fp16 and left every later gradient at zero
(above). Training now uses bf16 autocast on CUDA, which has the range of fp32 and needs no
loss scaling; other devices train in fp32. A step whose loss or gradient is not finite is
skipped, leaving the weights, optimizer, schedule and EMA as they were, and is logged as
`nonfinite_step` in `logs/events.csv` with the loss terms or parameter at fault and its clips
(`episode file:start frame`). After `optimization.max_nonfinite_steps` (10) skipped steps in a
row the run saves a crash checkpoint and stops.

## Resuming and warm starts since Phase 2

The learning rate of each optimizer step comes from the config and the number of steps taken,
so a run trains at the rate its config states; settings can no longer be changed mid-run. A
resume continues exactly what the run started. If a setting that changes the training differs
from the checkpoint's (model, data, optimizer, schedule, loss, seed), the trainer lists the
differences and stops before writing anything. Where and how the run executes may change: the
device, the data path, loader workers, logging and checkpoint cadence. To continue with other
settings, a new run starts from the checkpoint's weights (`--set run.init_from=<checkpoint>`)
with a new optimizer and schedule, and `logs/events.csv` records the checkpoint's path and
sha256 as a `warm_start` event. Older configs keep a `runtime_overrides` section, which loading
ignores; no recorded run used it.

## Temporal core since Phase 2

The RWKV core is RWKV-4-style with data-dependent decay. Each layer carries its time-mixing
shift, WKV memory and channel-mixing shift between calls, so one call over a clip and one
`step` per frame give the same result, and the memory is a normalised average kept in fp32.
The two paths agree to 2e-6 in fp32. Under bf16 they agree only to rounding (0.008 on features
of standard deviation 1), because GPU rounding depends on how frames are batched, so driving
runs in fp32. The April run was trained on the earlier core. Its checkpoints and the model scores above can
only be reproduced from tag `v0.2.0-p1`.

The GRU reference has the same residual shape: each of its layers reads layer-normalised
features and adds its output to them. A plain stack of six GRU layers stopped responding to its
input in the [short run](#short-run).

`model.encoder.core: rwkv7` selects RWKV-7 (`src/models/rwkv7_encoder.py`), built as the
official RWKV-LM blocks (`RWKV_Tmix_x070`, `RWKV_CMix_x070`) with their initialisation. Its
memory is a matrix per head of 64 channels, updated by the generalised delta rule, so it can
replace what it holds for a key rather than only let it fade; later layers mix in the first
layer's values. It carries three tensors per layer like the RWKV-4-style core and runs the
memory one frame at a time in fp32. With the same random weights (3 layers, 2 heads, 24
frames), its output equals the reference's plain-PyTorch implementation exactly over a clip,
and to 8e-6 one frame at a time. At the short-run width its core has 11.8 million parameters,
against 12.4 million for the RWKV-4-style core and 5.3 million for the GRU. Its gate runs are
under [RWKV-7 core](#rwkv-7-core).

## Visual stem since Phase 2

The stem is an ImageNet ResNet-18 with frozen batch norm. Its 512×5×30 feature map is projected,
given position codes and pooled by 8 learned queries into one token per frame. Only the last 4
frames of a clip train the backbone; the pooling head trains on every frame. The backbone trains
at a tenth of the learning rate (`optimization.backbone_lr_scale`): at the full rate, its frozen
batch norm turned small weight changes into an offset shared by every frame, and the frame
tokens stopped varying ([short run](#short-run)).

`python scripts/probe_turn_tokens.py --episodes 250` trains the pooling head alone to tell left
turns from right on 2,522 frames of 159 testbed episodes, with mirrored copies, and scores 636
frames of 40 held-out episodes. Scores are balanced accuracy, from one split and seed. The pass
mark, set before the run, is 80% and above the random-backbone control.

| Backbone | Training | Held-out |
| --- | --- | --- |
| ImageNet ResNet-18 | 100.0% | 92.5% |
| Random ResNet-18 (control) | 97.3% | 85.8% |

The golden subset has too few turning frames for this check (515 frames from 10 windows).

## Checkpoint selection since Phase 2

`best.pt` is the checkpoint with the lowest validation `control_score`, the mean of two errors
pooled over every scored frame: `moving_control_mae` on moving frames and `turn_control_mae` on
moving frames with |expert steering| above 0.1. Each is the absolute error averaged over steering
and acceleration. Burn-in, stationary and excluded frames do not count. The total loss chose the
April checkpoint, whose steering was no better than always steering straight.

## Learning gates (Phase 2)

### Overfit one window

`make overfit` trains the short-run model on the golden window with the most turns
(`episode_8de8b157`, 16 clips), with augmentation, dropout and weight decay off, and scores
steering on its moving frames after burn-in. The window's steering is replaced by its 5-frame
running median ([steering spikes](LABELS.md#steering-spikes)): scored against the raw labels,
a perfect fit to the median would still have an error of 0.0246, above the 0.02 limit.

| Steering labels | Learning rate | Backbone | Steps | Steering MAE | Turn sign | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Raw | 3e-4, constant | ×1 | 2,000 | 0.0715 | 98.1% | failed |
| 5-frame median | 1e-4, cosine to 1e-5 | ×1 | 2,700 | 0.0195 | 100.0% | passed |
| 5-frame median | 1e-4, cosine to 1e-5 | ×0.1 | 1,400 | 0.0159 | 100.0% | passed |

Run on MPS in fp32: 35 minutes for the first passing run, 18 minutes for the last. With the
sparse target point and the situation head, the short-run default since, the gate passes at
step 1,500 (0.0150, 100.0%, 20 minutes), and with the RWKV-7 core as well at step 1,000
(0.0192, 100.0%, 12 minutes). On the raw labels, training on the control loss alone, a learning rate of 1e-4,
no gradient clipping and the GRU core each stayed far from the limit after 600 steps (best
0.0768).

### Short run

`configs/short_run.yaml` trains for one epoch of the testbed's 880 training episodes, in clips
that do not overlap: 5,333 steps of 8 clips of 32 frames, learning rate 3e-4 after 50 warm-up
steps with cosine decay to 3e-5, in fp32 on MPS (2 to 2.5 hours). Validation every 500 steps on
its first 200 batches picks `best.pt`. `scripts/evaluate_offline.py` then scores `best.pt` and
`latest.pt` on all 219 held-out episodes (249,792 frames, 33,921 weighted turn frames). Set
before the runs, a run passes if its turn MAE is below the target-point lookup's and rises by at
least 10% when each clip is given another clip's images with its own command, speed and target
point (`other_clip_turn_error_ratio` of 1.10 or more).

Scores of `best.pt` on turn frames, training weights:

| Run | Temporal core | Backbone learning rate | Token spread at the last validation | Turn MAE | Turn sign | On other clips' images | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `short_rwkv` | RWKV | 3e-4 | 3e-5 * | 0.1269 | 94.9% | not measured | blind |
| `short_rwkv_v2` | RWKV | 3e-5 | 0.65 | 0.1148 | 95.3% | ×1.49 | passed |
| `short_gru_v2` | GRU, plain stack | 3e-5 | 0.003 | 0.1241 | 95.1% | ×1.00 | failed |
| `short_gru_v3` | GRU, residual | 3e-5 | 0.65 | 0.1140 | 95.1% | ×1.50 | passed |
| Target-point lookup, 60 bearing bins | | | | 0.1548 | 94.9% | | |
| Per-command mean | | | | 0.2082 | 85.8% | | |

\* Measured after the run on 4 held-out clips; validation did not log token spread yet.

`short_rwkv`: at the full learning rate the backbone's features grew from 0.87 to about 100
within 250 steps while its weights moved only 1–7%, and mirroring the cameras changed the
steering by 0.0000. Its lead over the lookup came from the target point alone.

`short_rwkv_v2`: token spread rose from 0.31 at step 500 to 0.65, and `latest.pt` scores the
same (turn MAE 0.1148, ×1.51 on other clips' images). In an episode bootstrap (95%) the turn
MAE is 0.036 to 0.045 below the lookup's and 0.052 to 0.061 below its own on other clips'
images. Mirroring the cameras flips the steering on 2.0% of turns: the direction of a turn
comes from the target point, the amount of steering from the images. A k-nearest-neighbour
regression on the target point alone (8 neighbours over the training episodes' moving frames,
since then the evaluator's `target_point_knn` row) still scores 0.0989, 0.0135 to 0.0185
better than the model.

`short_gru_v2`: the health check failed it, with flat frame tokens from step 500, and its
output did not change at all with other clips' images or mirrored cameras. Its backbone moved
only 0.4–2%. The plain six-layer stack halved the frame-dependent part of its input at every
layer, the frame tokens drifted to a large shared offset that saturated its first layer, and
the trained core no longer responded to its input.

`short_gru_v3`, with the residual GRU core: token spread rose from 0.40 at step 500 to 0.65, and
`latest.pt` scores 0.1135, ×1.50 on other clips' images. In the episode bootstrap the turn MAE
is 0.037 to 0.045 below the lookup's and 0.052 to 0.062 below its own on other clips' images,
and the k-NN on the target point is 0.0127 to 0.0175 better. Mirroring the cameras flips the
steering on 3.1% of turns. The two cores score within 0.001 of each other here, from one seed
each and in models of different sizes (19.1 million parameters with the GRU core, 26.2 million
with RWKV); comparing them is left to Phase 3.

### Sparse target points and a situation head

In the runs above, a nearest-neighbour regression on the sliding target point alone beat every
model. Four more short runs changed one thing at a time: the sparse target point, which the
CARLA leaderboard gives agents ([target point](LABELS.md#target-point)), and a situation head
that predicts what only the images show (a red light, a stop sign, a vehicle or a pedestrian
in the path), with `loss.situation_weight` 0.1. Set before the runs, a run passes if its turn
MAE is below both image-free baselines built from its own kind of target point (the 60-bin
lookup and `target_point_knn`), rises by at least 10% on other clips' images, its situation
head beats a constant answer on each cause (the episode-bootstrap 95% interval of the AUROC
lies above 0.5) and its frame tokens stay healthy. Intervals come from 1,000 episode
bootstrap draws over the evaluator's per-frame file.

Scores of `best.pt` on turn frames, training weights, all four with healthy frame tokens
(spread 0.63–0.64 at the last validation):

| Run | Target point | Situation head | Turn MAE | 60-bin lookup | Nearest neighbours | Model minus nearest neighbours, 95% | On other clips' images | Result |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| `short_rwkv_head` | sliding | yes | 0.1141 | 0.1548 | 0.0989 | +0.013 to +0.017 | ×1.54 | failed |
| `short_rwkv_sparse` | sparse | no | 0.1139 | 0.1472 | 0.1184 | −0.008 to −0.002 | ×1.48 | passed |
| `short_rwkv_both` | sparse | yes | 0.1115 | 0.1472 | 0.1184 | −0.010 to −0.004 | ×1.61 | passed |
| `short_gru_both` | sparse | yes | 0.1149 | 0.1472 | 0.1184 | −0.007 to −0.000 | ×1.56 | passed |

`latest.pt` gives the same results, with turn MAEs of 0.1136, 0.1110, 0.1119 and 0.1105.
Situation AUROC of `best.pt` on the validation frames that count in training:

| Run | Red light | Stop sign | Vehicle in path | Pedestrian in path |
| --- | ---: | ---: | ---: | ---: |
| `short_rwkv_head` | 0.975 | 0.902 | 0.957 | 0.788 |
| `short_rwkv_both` | 0.974 | 0.873 | 0.955 | 0.814 |
| `short_gru_both` | 0.979 | 0.937 | 0.961 | 0.862 |

The pedestrian intervals are wide (0.57 to 0.98 across the runs): 936 validation frames have
one in the path.

The models' turn MAE barely moves between runs. What the sparse point changes is the
image-free baseline: from a sparse point the nearest-neighbour regression scores 0.1184, not
0.0989, and the models beat it. The situation head learns its causes, but with the sliding
point the nearest-neighbour regression still wins; added to the sparse point, it raised the
model's reliance on the images from ×1.48 to ×1.61. With one run each, RWKV and GRU tie. Short
runs now use the sparse point and the situation head (`configs/short_run.yaml`).

### RWKV-7 core

`short_rwkv7` repeats `short_rwkv_both` with `model.encoder.core: rwkv7`, under the same pass
bar. It trained in 2 hours 16 minutes, and its token spread was 0.61 at the last validation.

| Checkpoint | Turn MAE | Model minus nearest neighbours, 95% | On other clips' images | Red light | Stop sign | Vehicle in path | Pedestrian in path | Result |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `best.pt` | 0.1125 | −0.009 to −0.003 | ×1.67 | 0.980 | 0.893 | 0.965 | 0.863 | passed |
| `latest.pt` | 0.1092 | −0.012 to −0.006 | ×1.67 | 0.981 | 0.906 | 0.965 | 0.868 | passed |

The last four columns are situation AUROCs. Its turn MAE is within 0.003 of the RWKV-4-style
core's (0.1115 and 0.1110). It relies on the images a little more (×1.67 against ×1.61), and its
AUROC is higher for every cause. From one run of each, these differences cannot be told apart
from what a different seed would give. Short runs now use the RWKV-7 core, the current RWKV
design; `core: rwkv` and `core: gru` remain for comparison.

## Driving (Phase 2)

`scripts/run_learned_agent.py` drives `agents/learned/learned_agent.py` in a synchronous world
at 20 Hz, the rate the expert recorded at. The agent steps the model once per frame of the rate
it trained at, every second tick, with the state carried from the frame before, in fp32, and
repeats its control on the tick between. It takes each camera's image of the tick by frame id
and prepares it as the training cache did (resize, JPEG round trip, normalisation); a test
checks the result equals the dataset's sample bit for bit. Before this, the model stepped at
20 Hz, its images skipped the JPEG round trip (up to 8 grey levels from training's), it took
the newest image whatever its frame, and a throttle of 0.15 was added below 5 km/h.

### Agent check: 500 ticks in Town01

`short_rwkv7` `best.pt`, seed 0, 50 vehicles and 30 pedestrians (`--max-steps 500 --seed 0
--log`). All 500 frames were consecutive and the model stepped on 250. On those ticks
`run_step` took 61 ms at the median and 83 ms at the 95th percentile: 28 ms waiting for the
cameras, 32 ms preparing the image and stepping the model (0.7 s for the first step). A 10 Hz
frame allows 100 ms. Two faults of the runner showed first: it planned the route from the
ego vehicle's position before the world had ticked, when a new actor still reads (0, 0, 0), so
every target point pointed hundreds of metres away; and it destroyed vehicles the traffic
manager still drove, which aborted the process.

The car, a Lincoln MKZ (see below), did not drive off. It crept at 1.3 km/h on average and covered 1.1% of its route, with
no collision. From a standstill the control head asks for little throttle (0 to 0.3), although
the waypoint head places the car 5–6 m ahead after 2 s and the speed head asks for about
9 km/h. On a clear road, the same frame gives an acceleration of 0.06 at 0 km/h, 0.21 at 5 km/h
and 0.34 at 10 km/h: the inertia problem of behaviour cloning (Codevilla et al., 2019), where a
model that sees the car standing still predicts it stays still. A third of the testbed's usable frames stand still. On
97% of them the expert brakes (mean acceleration −0.47), even within a second of pulling away
(median −0.50), and standing frames count a tenth in the loss.

### On the vehicle the expert drove

The runner spawned a Lincoln MKZ, but the expert recorded every episode with a Tesla Model 3, so
the throttle and steering the model learned belong to another car. On the Tesla
(`--max-steps 1500 --seed 0 --log`), the car drove 35% of its route in 75 s with no collision,
at 11.9 km/h on average and up to 38 km/h, with `run_step` at 64 ms at the median and 88 ms at
the 95th percentile. Its situation head passed 0.5 for a red light 1.7 s before CARLA counted
the car as at the light. It slowed from 18 km/h, stopped, waited 29 s and moved off 0.35 s
after the light turned green, at a throttle of 0.46 from the first frame of green.

Before that it stood for 4.5 s after spawning, at a throttle of 0.10–0.15, too little to move
the car. The expert drives off within 0.5 s of every episode start and the first 8 frames of a
training clip only warm the state, so no frame the model trained on starts from rest with an
empty state.

### Waypoint check

TransFuser, TCP and InterFuser drive with waypoints: a controller follows the predicted
positions and starts the car when they lie ahead of it. Set before the run, the waypoint head of
`short_rwkv7` `best.pt` would be ready to drive a controller if all of these hold:

1. On moving frames, its average displacement error (ADE) is below both image-free baselines,
   constant velocity straight ahead and the 8 nearest training frames by target point and
   distance covered in 2 s at the current speed. The 95% interval of the difference must lie
   below 0.
2. At standstill, its predicted distance after 2 s separates pulling away (2 m or more in 2 s)
   from waiting (under 0.5 m). The AUROC's interval must lie above 0.5.
3. It predicts under 0.8 m after 2 s, the 0.4 m/s below which TransFuser's controller brakes,
   on at least 95% of waiting frames.

Frames are those of the offline evaluation (held-out split, after burn-in, counting in
training), with waypoints at 0.5, 1.0, 1.5 and 2.0 s and horizons past the end of the recording
left out. On moving frames, in metres:

| Predictor | ADE | Error at 2 s | ADE on turn frames |
| --- | ---: | ---: | ---: |
| `best.pt` | 0.637 | 1.219 | 0.817 |
| `latest.pt` | 0.633 | 1.209 | 0.811 |
| Constant velocity | 1.076 | 2.172 | 2.151 |
| Nearest neighbours | 0.496 | 0.969 | 0.595 |

`best.pt` minus constant velocity: −0.46 to −0.41 m; minus nearest neighbours: +0.12 to
+0.17 m. At standstill, on 4,945 frames pulling away and 71,991 waiting:

| Checkpoint | Go/stop AUROC, 2 s distance | Go/stop AUROC, control head | Waiting under 0.8 m | Pulling away over 0.8 m |
| --- | --- | --- | --- | ---: |
| `best.pt` | 0.692 (0.670 to 0.711) | 0.696 (0.673 to 0.717) | 87.8% (85.1 to 90.4) | 43.7% |
| `latest.pt` | 0.680 (0.658 to 0.700) | 0.697 (0.674 to 0.719) | 90.5% (88.1 to 92.6) | 41.0% |

Both checkpoints fail conditions 1 and 3 and pass 2. The head beats driving straight on but not
a lookup that knows only the target point and the speed. Its go/stop signal is no better than
the control head's, and a controller on it would set off on about one waiting frame in ten. In
this run the waypoints are an auxiliary loss at weight 0.5. The agent keeps the control head,
which drives on the expert's car; the same check repeats on the MVP model.

### One route to its destination

`short_rwkv7` `best.pt` on the same route, with up to 6,000 ticks. The car reached its
destination at tick 4,336 (217 s): 99.6% of the route at 11.6 km/h on average, with `run_step`
at 67 ms at the median and 95 ms at the 95th percentile. It waited 29 s at a red light and 13 to
29 s behind other vehicles three times, and set off again each time. The red light was the only
one it reached first in line; the red-light head gave 0.97 for it. At one of the seven lights
on its lane CARLA did not report the light, so its state is unknown; the head gave 0.88 just
before it and the car crossed at 15 km/h.

Red lights here count only for the car's own lane: a light is run when the car passes that
lane's stop waypoint on red and drives on into the junction. The stop waypoint is CARLA's
(`get_stop_waypoints`), the line the labels measure stops from; in Town01 it lies 6.5 to 13 m
before the junction, where the CARLA leaderboard draws its line. CARLA reports a car at a
traffic light whenever its body overlaps that light's trigger zone, and a car leaving a junction
after a turn can brush the zone of the light for oncoming traffic, which may be red while its
own is green. The runner logs only the light CARLA reports.

It collided six times (92 events of the collision sensor). Four were single hits at 2.9 to
8.6 km/h, a few seconds after braking from 25 to 37 km/h. Two were sustained contacts that
pushed against what was ahead: one set off at a throttle of 0.51 after waiting 14 s behind it
(59 events over 5.8 s), the other arrived at 3.3 km/h (29 events over 2.5 s). In five of the six
the situation head gave 0.96 to 1.00 for a vehicle in the path, so the model saw something ahead
and still braked late or set off. The runner did not yet log what each collision hit.

### Starting from rest

Each drive stood still for about 5 s after spawning. Without a model, the expert's car held at a
constant throttle from rest moves after 6.1 s at 0.12, 1.3 s at 0.3, 0.95 s at 0.4, 0.75 s at
0.5 and 0.40 s at 1.0, however long it settled first. The model holds 0.10 to 0.18. The expert,
at the start of every training episode, stands for 5 frames at full throttle and moves at the
sixth. No training frame showed this: clips started at an episode's first frame 34.5 times an
epoch in 42,671, and the first 8 frames of every clip were burn-in, which warms a state that a
drive does not have at its start either.

The change: a clip that starts an episode counts from its first frame, since it begins from the
empty state a drive begins from; clips that start mid-episode keep their burn-in. Each
episode's first clip is drawn as often as one of its fixed clips on average, 865 times an epoch
(2.0%). The runner now lets the car settle for 10 ticks before the agent's first step, as the
expert's collector does, and logs what each collision hit. The evaluator reports the model's
acceleration on the validation clips that start an episode, until the car first moves.

Set before the run, `short_rwkv7_start` (`short_rwkv7` with this change, same configuration,
seed and data) passes if (1) on those frames the episode-bootstrap 95% interval of its median
acceleration lies above 0.4, which moves the car within 1 s; (2) in Town01 with seed 0 the car
passes 1 km/h within 1.0 s of the agent's first step; and (3) it still passes the bar of the
short runs above. It trained in 2 hours 13 minutes, with a token spread of 0.62 at the last
validation.

| Checkpoint | Median acceleration from rest, 95% | 0.4 or more | Turn MAE | Model minus nearest neighbours, 95% | On other clips' images | Red light | Stop sign | Vehicle in path | Pedestrian in path |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `short_rwkv7` `best.pt` | 0.195 (0.170 to 0.216) | 1.5% | 0.1125 | −0.009 to −0.003 | ×1.67 | 0.980 | 0.893 | 0.965 | 0.863 |
| `short_rwkv7_start` `best.pt` | 0.435 (0.398 to 0.471) | 56.1% | 0.1108 | −0.011 to −0.005 | ×1.66 | 0.978 | 0.896 | 0.957 | 0.819 |
| `short_rwkv7_start` `latest.pt` | 0.444 (0.417 to 0.475) | 59.0% | 0.1126 | −0.009 to −0.003 | ×1.63 | 0.979 | 0.883 | 0.962 | 0.833 |

The acceleration is over 1,095 frames from 219 episode starts, where the expert's median is
1.00. In the drive, `best.pt` gave 0.49 to 0.51 from its first step and the car passed 1 km/h
after 0.80 s (`short_rwkv7`: 0.12 to 0.30, and 4.8 s). Judged on `best.pt`, as every bar here
is, the run passes (2) and (3) and misses (1) by 0.002: its interval starts at 0.398.
`latest.pt` clears it. The waypoint check gives the same results as `short_rwkv7`'s (ADE 0.639,
go/stop AUROC 0.661 against 0.692, waiting frames under 0.8 m 90.9%). One run of each cannot
tell the other differences from a change of seed.

On the full route, `short_rwkv7_start` reached its destination at tick 4,345: 99.4% of the
route at 11.7 km/h, with `run_step` at 64 ms at the median and 88 ms at the 95th percentile. It
stood 27, 17, 31 and 6 s behind vehicles, with the red-light head at 0.80 to 0.97, and set off
each time at 0.43 to 0.46 without a collision. It reached no red light first in line. It
collided once: it slowed from 23 to 10.5 km/h in a junction, sped up again and pushed against a
bus for 3 s (73 events), then touched a car; the vehicle-in-path head gave 0.99 throughout. No
bar covered collisions or red lights.

### Red lights

A second drive of `short_rwkv7_start` `best.pt` on the same route saved the image the model
saw at each step and logged the traffic light ahead in the car's lane, with its state and the
distance to its stop waypoint. With the same model and seed, it took a different course from the
drive above and did not collide, so one drive per model is a noisy measure.

It met six red lights. At four it stopped behind a vehicle, 7.6 to 19 m short of the stop
waypoint, with the vehicle-in-path head at 0.99 to 1.00. At the other two no vehicle was ahead
(0.04 to 0.53), and the red-light head gave 0.7 or more from about 25 m and 0.98 to 0.99 at
the stop waypoint. At one the car eased off from 26 km/h, braked from the stop waypoint and
stopped 1.9 m past it, 6.6 m short of the junction, until the light turned green; the label
audit counts a stop up to 4 m past the stop waypoint as in place. At the other it eased off the
throttle from 0.35 to 0.08 and drove through at 17 km/h without braking. At 20 m a light covers
a few pixels of the 160 × 960 image, yet the head found both. The model sees the light, but
brakes for it alone less readily than for a vehicle.

On validation frames approaching a red light at over 2 m/s with no vehicle in the path, the
expert brakes on 26% and the model on 17% (`short_rwkv7`: 19%). With a vehicle as well, the
expert brakes on 43% and the model on 31%. The expert does stop for the light alone: on 71% of
the frames it stands at a red light, no vehicle is in its path.

### Starting from rest, second run

`short_rwkv7_start` still weighed the frames at rest that start an episode at 0.1, as it weighs
every standing frame. The change: from an episode's first frame until the car first moves, the
frames weigh 1.0, about 4,300 frames an epoch; sampling is unchanged. Validation's control score
counts them with the moving frames, about 170 among about 25,000, so the choice of `best.pt`
leans slightly toward starting. Set before the run, `short_rwkv7_rest` (`short_rwkv7_start`
with this change, same configuration, seed and data) is judged on `best.pt` by the same three
conditions. It trained in 2 hours 15 minutes, with a token spread of 0.62 at the last
validation.

| Checkpoint | Median acceleration from rest, 95% | 0.4 or more | Turn MAE | Model minus nearest neighbours, 95% | On other clips' images | Red light | Stop sign | Vehicle in path | Pedestrian in path |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `short_rwkv7_rest` `best.pt` | 0.667 (0.616 to 0.692) | 76.2% | 0.1111 | −0.011 to −0.004 | ×1.61 | 0.979 | 0.925 | 0.959 | 0.901 |
| `short_rwkv7_rest` `latest.pt` | 0.684 (0.650 to 0.699) | 77.4% | 0.1106 | −0.011 to −0.005 | ×1.63 | 0.978 | 0.915 | 0.961 | 0.896 |

It passes all three. In the drive, `best.pt` gave 0.71 to 0.74 from its first step and the car
passed 1 km/h after 0.55 s. The waypoint check gives the same results as before (ADE 0.638,
go/stop AUROC 0.670, waiting frames under 0.8 m 88.4%).

Each change toward starting made the model readier to move where the expert waits. On standing
validation frames where the expert waits, its acceleration is above 0.3 on 2.0% (95% interval
1.5% to 2.6%; `short_rwkv7_start` 1.6%, `short_rwkv7` 0.4%), and on 2.1% of those with a vehicle
in the path (2.0%, 0.7%); where the expert pulls away, on 33.8% (30.8%, 26.8%). The red-light
head gives over 0.5 on 28.7% of standing frames without a red light (36.8%, 18.8%). Approaching
a red light with no vehicle in the path, the model brakes on 18% of frames, the expert on 26%.

On the full route it reached its destination at tick 4,455: 99.4% of the route at 11.2 km/h,
with `run_step` at 69 ms at the median and 100 ms at the 95th percentile. It ran one of the
seven lights on its lane, at 21 km/h without braking, with the red-light head at 0.48 to 0.74;
it passed the others on green or yellow. It collided eight times (357 events).
In the junction past that red light it hit something static at 18 km/h and stayed against it
for about 15 s (five contacts, 106 events). Braking behind a vehicle, it accelerated again at
0.51 from 8 km/h as the vehicle-in-path head fell from 0.99 to 0.46, and hit it at 9 km/h.
After standing 20 s behind a vehicle, it set off at 0.59 to 0.65 and touched it at 1 km/h. In
a left turn at 16 km/h it hit a vehicle, with the vehicle-in-path head fallen from 0.99 to
0.47, and stood against it for 12 s (239 events). It set off at 0.16 to 0.65 after standing, as
`short_rwkv7_start` did in its two drives (0.22 to 0.59). With one collision and none in those
two drives, one drive cannot show whether the change made it worse; no bar covered collisions or
red lights.

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
