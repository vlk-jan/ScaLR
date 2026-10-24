# V10 scope: should there be another ScaLR pretraining for DRIFT, and what would it change?

*2026-10-17. Read-only scoping, nothing submitted. Main source:
`~/projects/DRIFT/docs/BACKBONE_VALUE.md` (cited as BV §n). Probe outputs:
`/mnt/personal/vlkjan6/scratch_wi_probe/v24{,_halfm}/analysis_wedge60*.txt`.*

**Recommendation in one line.** Do not launch a 5-day DINOv3-distillation v10
now. Nothing measured says a better distillation would help DRIFT's compression
heads, and two training runs say the current one hurts them. First run the
cheap experiments in §4. A full v10 is triggered only by the gate in §3, run on
a pilot checkpoint of a *changed objective*, not by a re-run of the v9 recipe
on v2.4.

## 1. What DRIFT needs from the backbone, and where v9's objective works against it

DRIFT reads a frozen WaffleIron. It mean-pools the tokens into 1 m BEV cells
(one view, `dim_proj [2]`) and realigns them onto the 0.1 m property grid next
to the height-map geometry channels (`geom4`).

**What it needs:**
1. **A scene-level compressibility prior that transfers to a new session.**
   The head takes "where" from the height map and "how compressible is this
   ground" from the backbone.
2. **Sub-metre rigid-vs-vegetation cues**, such as a trunk standing in grass.
   This is optional: the 1 m case is already solved without the backbone.

### Established (measured, with source)

| # | Fact | Source |
|---|---|---|
| F1 | The label is about 90 % geometry. `geom4` alone gives within-frame R² 0.899, and the height map alone 0.882. | BV §1 |
| F2 | On top of `geom4`, in a session-held-out ridge: v9 block 36 MAE 0.0410, 16 hand stats 0.0390, random WI block 0 0.0379. v9 block 6 does best (0.0345). | BV §1 |
| F3 | A flexible readout breaks on the final layer of every pretrained backbone across sessions: MLP + geom4, v9 L36 MAE 0.1377 vs random L36 0.0477. It holds on random and shallow taps. So the final layer carries session-specific structure. | BV §1 (MLP paragraph) |
| F4 | Head knockout. With the backbone features zeroed, within-frame correlation is unchanged (0.919 vs 0.911) and only the level moves. Features from 6 s earlier serve almost as well. The backbone acts as a scene-type scalar. | BV §2 |
| F5 | **Training runs: frozen random WI beats WI_v9.** v23_rand − v23_ctl: −6.6 mm pooled [−14.6, −1.6], −17.8 mm far. v24combo (random WI, v2.4) − v23_ctl: −8.2 mm pooled. v24combo ties v23_rand on every band, so the v2.4 data is neutral. | BV §9, §10.3 |
| F6 | The teacher is a low ceiling. DINOv3 + geom4 is the *worst* row (0.0501, within-frame R² 0.866, below geom4 alone). A perfect student would not help. | BV §1 |
| F7 | Teacher coverage is not the compression problem: 91 % of label cells have a DINOv3 target. | BV §5 |
| F8 | Viewpoint encoding is architectural. A random WI predicts range at R² 0.84 and bearing at 0.78, more than v9 does (0.71 / 0.65). The features restate per-cell intensity (R² 0.99-1.00), density and z-spread. Effective rank of v9 L36 is 9.2/768. | BV §3 |
| F9 | Intensity reaches the backbone 7 sd off: mean 7.30, sd 0.91 under DRIFT's v1 constants, against roughly N(0,1) in pretraining. Rescaling it de-collapses the features (rank 9.2 → 17.6) but worsens L36 readout (0.0410 → 0.0516). It improves L6 (0.0334, the best row). | BV §4 |
| F10 | Under the pretraining projection (3 views, 0.5 m), v9 L36 separates trunk from tall vegetation at AUC 0.95-0.96. At 0.2 m it reaches 0.961, but production's 1 m pooling drops it to 0.86. Random WI reaches 0.93 per point. Compression barely moves: the v15 fullproj pair is +1.1 mm. | BV §7 |
| F11 | v24_proj (pretraining projection + rig intensity, v2.4) is **+8.8 mm worse in the near field** [+2.9, +18.7]. It fits the training-scene val fold better (corr_within 0.93 vs 0.86). More backbone fidelity means more session memorisation. | BV §10.1 |
| F12 | Trunks at 1 m are solved by labels, not by the backbone. v24combo (random WI + rigid labels) reaches rigid_hr_cone 0.003, and v23_rigid 0.011. The token readout on pretrained tokens reaches 0.012. | BV §10.2-10.3 |
| F13 | Earlier pretrained-vs-pretrained swaps were inert. decorr = v6 on every head metric. v6 "cured collapse" and bought DRIFT nothing. | memory `v6-backbone-inert-for-drift` |

### Hypotheses (not established)

- **H1, intensity:** a readable intensity channel makes features
  session-specific, and that drives F11. The prime suspect after F5 cleared the
  data and the v15 pair put the projection at +1 mm. Readout support: the MLP
  + geom4 row at L12 holds for v9 (0.0428) but breaks for v9fix (0.0605)
  (`analysis_wedge60_mlp2.txt`). **Test queued:
  v24_proj_legacy**, the proj arm with `--intensity_norm v1_legacy`, paired
  against v24_proj. Batch: `~/projects/DRIFT_fine/slurm/v24.batch`,
  `INORM=v1_legacy`.
- **H2, objective:** image-appearance targets are themselves
  session-specific (F6). Any student that matches them better transfers worse
  (F3, F11). If H2 holds, no teacher-layer, coverage or epoch change rescues
  distillation for compression.
- **H3, shallow taps:** blocks 6-12 carry a transferable 4-5 mm. Under the
  pretraining projection, v9 L6/L12 beat the hand stats by 4.5-5.4 mm in 6-7
  of 7 sessions (BV §7, readout only). No head has ever read a shallow tap, so
  this is unproven in training.
- **H4, stage 3:** training WI on DRIFT's own loss (unfreezing) would beat a
  frozen random WI. Never measured. `--unfreeze_backbone` exists in DRIFT
  `train.py`, and a b32 frozen/unfrozen pair was scripted on 2026-08-24, but I
  found no recorded verdict.
- **Caveat on F5:** there is one random seed. BV caveats list this; it has
  not been checked.

## 2. Candidate changes to the ScaLR recipe

Paths are in ScaLR unless marked DRIFT.

| # | Change | Motivating finding | Concrete change | Expected effect | Cost | Risk |
|---|---|---|---|---|---|---|
| C1 | **Abandon distillation for compression: keep the frozen random WI.** | F5, F6, F2 | None in ScaLR. DRIFT already ships `checkpoints/WI_random_seed0.pth` (v24combo). | Holds the current best. | 0 | Single seed (check with C1b). A random WI still costs 64-85 % of DRIFT's step for features 16 stats nearly match. |
| C1b | Random-seed spread | F5 caveat | DRIFT: `probe_backbone_value.py --random_seeds 0 1 2` (dump only) | Error bar for "random". | ~1-2 h on 1 GPU (gpufast) | None |
| C2 | **Intensity: replace it with `constant` (or a per-frame z-score).** | F9, H1, F11 | ScaLR: `input_features: ["constant","xyz","radius"]` keeps `size_input 5`. The `constant` branch already exists in `datasets/pc_dataset.py::prepare_input_features`. Per-frame z-score instead: a new branch next to it. DRIFT: add the same branch to `datasets/pc_dataset.py::prepare_input_features`, which today raises on `constant`. `drift_ros2` must mirror it. | Removes the session-specific material channel and the rig-constant trap entirely. | Pilot ≤ 24 h. DRIFT edit ~3 lines plus a lockstep ROS change. | Loses a real material cue. The muted channel read out well at L36 (F9), so it may cost compression. Only worth it if v24_proj_legacy confirms H1. |
| C2b | Intensity augmentation (per-frame gain/offset jitter) | Same | `datasets/im_pc_dataset.py`: add a transform after `prepare_input_features` (line ~394), before the point augmentations. Only the intensity column: gain U(0.5, 2), offset ±1 sd. | Keeps the cue, kills absolute-level reliance. | Pilot ≤ 24 h | Untested whether jitter is enough. Weaker than C2. |
| C3 | Teacher choice or layer (another ViT, intermediate layer, a smaller DINOv3) | F6 | `image_backbone.images_encoder` / `feat` in the config. `models/image_teacher.py` already supports DINOv2/v3 S-H and `forward_get_last_n`. | **Probably nil for compression** (H2). Any image teacher encodes appearance. | Teacher dump: `DRIFT/scripts/analysis/probe_teacher_cells.py`, run in the ScaLR env, ~2 h. Scoring a candidate teacher with `analyze_backbone_value.py` BEFORE any pretraining is cheap. | Spending a pilot on it before the teacher row passes. Rule: a teacher whose own + geom4 row does not beat random WI is not distilled. |
| C4 | Teacher coverage beyond the frustum (rear and lateral via temporal reprojection) | F7 says not needed for compression. The ±60° lidar wedge means nothing is rear-sensed anyway. | Dataset-layer project (`docs/objective.md`: needs pose, which the loaders lack) | ~0 for compression | Days of loader work | High effort, no motivating number. **Drop.** |
| C5 | **Session-adversarial head (GRL)** | F3, F11 | `utils/distiller.py`: copy the `PositionInvarianceHead` + `gradient_reversal` pattern. Use a `SessionHead` predicting session id (cross-entropy) from pooled frame features. Loaders must return a session id: HELHEST v2 has `manifest.json` sessions; other members get one id per dataset. Config block `session_invariance: {enabled, weight, grl_lambda}`. | Directly attacks session-specificity. | ~1 day code + pilot | The teacher target is session-specific (H2), so the GRL fights the main loss. The posinv pilot (GRL on position) did not remove position (memory `wi-features-encode-viewpoint`). Low prior. |
| C6 | Stronger point augmentation | F8, F11 | `im_pc_dataset.py` lines 132-144: today z-rotation, flip, scale 0.1, xy-translation 10 m. Add point dropout/density resampling (range-dependent), z-jitter, and C2b. | Weaker per-cell density/viewpoint shortcut | Small | The teacher pairing must survive dropout (`pairing_points` remap already exists for `LimitNumPoints`). |
| C7 | Cross-session or temporal positives (same place, other pass) | F3 | Needs pose plus correspondences (`docs/objective.md`, "considered, not implemented"). Cross-session would need GNSS/map registration across days. | The objective that matches the need best | Weeks | Highest effort. Only after cheaper evidence says a trained backbone can beat random. |
| C8 | Collapse fixes (`loss.collapse_weight`, cosine loss) | F8 (rank 9) | Config only: `loss.collapse_weight > 0`. Both knobs already exist. | Healthier features | Free in a pilot | **F13: de-collapse (v6, v9fix) never moved DRIFT.** F9: de-collapsing v9 worsened L36. Do not treat it as a lever. Log it as a health metric only. |
| C9 | Match the projection to DRIFT's readout | F10, F11 | Either pretrain with `dim_proj [2]`, 1 m cells (`grid_shape [[128,128]]`, fov ±64), or keep pretraining as is and have DRIFT run `configs/models_geom_fullproj_halfm.yaml`. | Compression: ~0 (v15 pair, +1 mm). Trunks: 0.86 → 0.96 AUC, but labels already solve trunks at 1 m (F12). | Free on the DRIFT side | Decided by v24_proj_legacy: if intensity is cleared, the projection is the suspect for +8.8 mm. |
| C10 | **Geometry- or DRIFT-native objective instead of image distillation** | F1, F2, F6 | (a) Stage 3, no ScaLR change: DRIFT `--unfreeze_backbone N --backbone_lr_scale 0.1` from random WI. (b) If a ScaLR-side pretext is still wanted, make it predict what transfers: per-cell height map / z-spread / occupancy of a future or masked sweep. That is a new loss in `utils/distiller.py` with no teacher. Also turn the proprio aux to v2 labels (`helhest_v2_labels` is `ignore` in v9). | (a) is the direct test of H4. | (a) one DRIFT arm. Unfreezing OOMs at b64, so b32 (commit "Add the batch-32 frozen/unfrozen matched pair"). | (a) can memorise sessions like F11. It must be scored paired on held-out sessions. (b) is a new research objective, not a recipe tweak. |
| C11 | Shallow tap / truncated WI (read block 6-12) | F2, H3, BV §6 | DRIFT side: needs new code. `truncate_wi_depth.py` and the k12 config were removed in the 2026-09-11 cleanup. ScaLR side: nothing, or pretrain a shallow net (depth 12). | Block 6 reads out best, at 1/6 of the backbone cost. | DRIFT code + one arm | Readout-only evidence (H3). The training-run test is pending. |
| C11b | Data: v2.4, epochs | memory `v2-4-impact-on-scalr` | `HELHEST_v2_4/scalr_keyframes.lst` with `--exclude` its held-out scenes. The corpus becomes 171,673 frames. **28 epochs** (~114 h at 0.985 steps/s). 32 epochs needs ~129 h, which is too tight for the 130 h limit. | ~0 by F5 (v2.4 was neutral for DRIFT) | 5 days of 3 H200s | Spending the quota for nothing if the objective is unchanged. |

**Bottom line of §2.** Only C2/C2b, C5, C6, C10 and C11 change the thing the
findings point at: session-specific appearance and intensity information in the
final layer. C3, C4, C8, C9 and C11b improve distillation fidelity, and F6, F11
and F13 say fidelity is not the lever. The evidence favours not pretraining for
compression (C1), with stage 3 (C10a) as the test of whether a *trained*
backbone can beat random at all. A ScaLR v10 is justified only if C10a or a
pilot from C2/C5/C6 clears the gate below.

## 3. A cheap gate that predicts DRIFT value before a 5-day run

**Lesson from the existing data.** The ridge + geom4 row at L36 *mispredicted*
the training run. It put v9 L36 at 0.0410, ahead of random L36 at 0.0420. Then
v23_rand beat v23_ctl by 6.6 mm. The **MLP row did predict it**: v9 L36 broke
to 0.1377 while random held at 0.0477. So the gate must use the session-held-out
**MLP readout at the tap DRIFT will read**, with ridge as a second leg.

**The bar** is random-init WI at the same tap and projection, under the same
intensity convention. It is not v9.

**Procedure.** About 3 GPU-hours plus CPU readout, on 1,119 frames, 36 scenes
and 7 session folds. Frames are reused so every existing row stays comparable.

1. **Pilot pretraining.** At most 24 h on 3 H200s (h200long, or the 40-min
   smoke `slurm/pretraining_robotics_v2_smoke.batch` for plumbing). That is
   about 6-7 epochs of the 171k corpus. `distill.py` writes `ckpt_last.pth`.
   DRIFT's `load_bare_ckpt` reads the `model_point` key directly, so no export
   is needed. Also probe a matched-length **v9-recipe pilot**. Otherwise
   "changed objective vs v9" is confounded with 6 epochs vs 32. If budget
   allows only one, the v9 final checkpoint plus random are the existing
   anchors.
2. **GPU dump.** Copy `DRIFT/slurm/audit/backbone_value_probe.batch`: gpufast,
   1 GPU, < 4 h, DRIFT env. Keep only the dump step:
   ```
   OUT=/mnt/personal/vlkjan6/scratch_wi_probe/v10_<name>
   mkdir -p $OUT; cp /mnt/personal/vlkjan6/scratch_wi_probe/v24/frames.txt $OUT/
   python3 -u scripts/analysis/probe_backbone_value.py \
     --rootdir /mnt/personal/vlkjan6/HELHEST_v2_4 --frames $OUT/frames.txt \
     --wi_ckpts pilot=/home/vlkjan6/projects/ScaLR/logs/pretraining/<run>/ckpt_last.pth \
                v9=./checkpoints/WI_v9_robotics_v2.pth \
     --all_taps_for pilot v9 --intfix_for pilot --random_seeds 0 1 2 --out_dir $OUT
   ```
   For a projection-matched run, add
   `--config configs/models_geom_fullproj_halfm.yaml`, as
   `slurm/audit/backbone_value_halfm.batch` does. The `<name>fix` row is the v2
   pretraining intensity convention, and the plain row is DRIFT main's legacy
   one. Read the row matching what the DRIFT arm will feed. For a C2
   (`constant`) checkpoint, DRIFT's loader needs the `constant` branch first;
   otherwise the dump raises.
3. **CPU readout** (`--mlp` is required):
   ```
   python scripts/analysis/analyze_backbone_value.py --dir $OUT --mlp
   ```
   Default `--wedge_deg 60`, session folds from `HELHEST_v2_4/manifest.json`.
4. **Optional, for C3 only.** Score a candidate teacher before distilling it:
   `scripts/analysis/probe_teacher_cells.py` (batch
   `slurm/audit/teacher_cells_probe.batch`, ScaLR env), then the same analyze
   step.

**Pass, all must hold:**

1. MLP + geom4 MAE at the target tap ≤ the best random seed's row − 2 mm. The
   ridge + geom4 row must also not be worse than random.
2. Paired per-session ridge MAE: better than random in ≥ 5/7 sessions. The
   analyze script prints "[k/7 sessions better]".
3. The MLP row does not break relative to its own ridge row. MLP MAE ≤ 1.3×
   ridge MAE is the F3 failure signature.
4. Logged, not gated: effective rank and mean |cos| (F8/F13 say they do not
   predict).

**Second stage, before the full v10.** If the pilot passes, run one DRIFT arm
on the pilot checkpoint with v24combo's recipe (`DRIFT/slurm/v24_combo.batch`
with `WI` swapped). Score it paired on v2.3 held-out against v24combo. The
probe is a predictor; issue 22 says backbones are decided by training runs.

## 4. Ranked plan

| Rank | Experiment | Cost | Depends on | Decision it makes |
|---|---|---|---|---|
| 0 | **v24_proj_legacy** (queued) vs v24_proj, paired | already queued | — | If legacy recovers most of the +8.8 mm, H1 holds and any future pretraining uses C2/C2b. If not, intensity is cleared, the projection on v2 is the cost (C9), and v24_proj's loss is not an input-convention issue. |
| 1 | CPU-only: extend the MLP readout (`analyze_backbone_value.py --mlp`, existing dump) to L6 for v9, v9fix and random. Today's file has MLP rows only at L12/L36. | ~1 CPU-h | — | Already in `analysis_wedge60_mlp2.txt`: **v9 L12 passes the gate** (MLP 0.0428 vs random L12 0.0478, ridge 0.0372, ratio 1.15). **v9fix L12 fails** (MLP 0.0605, ratio 1.6): the correctly normalised intensity breaks the flexible readout even at mid depth. That is independent support for H1. Fix the tap any later arm reads. |
| 2 | C1b random-seed spread (dump with `--random_seeds 0 1 2`, no new checkpoint) | ~2 GPU-h gpufast | — | Turns the "random" bar from one seed into a band. |
| 3 | **C10a stage 3**: v24combo recipe + `--unfreeze_backbone 12 --backbone_lr_scale 0.1`, from random WI, b32 config; also the matched frozen b32 control | 2 arms, ~2×27 h on 4 A100 | 2 (the bar) | **Can training the backbone on DRIFT's own loss beat frozen random on held-out sessions?** If NO: no pretraining of any kind is worth the quota; stop, and pursue cheaper inputs (C11, 16 stats). If YES: a trained backbone helps, which justifies one ScaLR pilot (rank 4). |
| 4 | One ScaLR pilot (≤ 24 h) with the objective change the evidence picks: C2 if rank 0 confirms H1, plus C6 (cheap). C5 only if C2+C6 fail the gate. Run the matched-length v9-recipe pilot alongside. | 2×24 h on 3 H200 | 0, 3 | §3 gate on both pilots. |
| 5 | DRIFT arm on the passing pilot checkpoint, paired vs v24combo | ~27 h on 4 A100 | 4 passes | Second-stage gate. |
| 6 | **Full v10**: 28 epochs on the v2.4 corpus (171,673 frames, `--exclude` v2.4 held-out incl. 0034/0035), recipe = the pilot that passed | ~114 h on 3 H200 (h200extralong) | 5 separated better than v24combo on pooled or far, near not worse | — |
| side | C11 shallow-tap arm in DRIFT (random WI or v9 L6/L12) | DRIFT code + 1 arm | 1 | Cheapest route to the 4-5 mm readout gain (H3) and a 6× faster backbone. Independent of any v10. |

**What triggers a full v10:** a ≤ 24 h pilot with a changed objective that
(a) passes the §3 gate against random WI at the tap DRIFT reads, and (b) beats
v24combo in a paired DRIFT arm on v2.3 held-out (segment CI separated, near
band not worse). A v9-recipe v10 on v2.4 data alone is ruled out by F5
(data neutral) and F11 (fidelity hurts).

**What stops the ScaLR line:** rank 3 says unfrozen/trained ≤ frozen random,
or rank 4's pilots fail the gate. Then the backbone question becomes "which
cheap input replaces it" (BV §6), and v10 is not run.

## Constraints a v10 batch must keep (from v9)

- Held-out scenes come from the manifest (`--exclude` in
  `scripts/write_scalr_keyframes.py`). The v9 batch's guard reads
  `held_out.scenes`, and v2.4 adds 0034/0035.
- `constant_bound` belongs to `$D/HELHEST → HELHEST_v1_5`. Re-measure it if the
  aux switches to v2 labels (C10b).
- Any input-convention change (C2) ships with its DRIFT loader branch and the
  `drift_ros2` switch. Checkpoints must carry the convention: DRIFT_fine
  records `intensity_norm` in `args.json`; add `constant`/`frame` modes there.
- 3 H200s, batch 12, lr 0.006, `--exclude=h01`. Never on dgx07.

## 5. What was launched (2026-10-19)

*Decision (user, 2026-10-19): skip the cheap DRIFT arms of §4 ranks 1-3 and go
straight to a changed objective, both arms at once.* (The 6-GPU per-user limit is per partition QoS: the H200 arms spend `extralongamd`'s, the A100 DRIFT arms `collaborator`'s on amdgpu, so the two do not block each other; user, 2026-10-19.) The
mechanism behind F3/F5/F11 is now measured (DRIFT `docs/BACKBONE_VALUE.md`
§11): the DINOv3 target is an appearance code of the recording session (one
pooled patch names the session at 98 %, 60 % of its variance is between
frames), blocks 24-36 of the student convert local geometry into that code,
and DRIFT's head memorises it. Two changes follow, and they are the two arms:

| arm | config | change against v9 |
|---|---|---|
| `v10_centred` | `configs/pretrain/WI_768_h200_v10_centred.yaml` | `loss.centre_teacher: True` -- every frame's teacher features are centred over the frame's paired pixels before normalisation, so only within-frame appearance contrast is imitated and a frame-constant component in the student can only cost cosine (`utils/distiller.py`, the gather loop). |
| `v10_geo` | `configs/pretrain/WI_768_h200_v10_geo.yaml` | centred, plus `geo_supervision`: a cooperative head on the block-36 tokens predicts each point's cell as seen CLOSE UP -- z_min, z_max − z_min, z_std, fraction of returns above z_min + 0.3 m, from the sweeps recorded while the robot drove up to it (`scripts/write_geo_targets.py`, `/mnt/personal/vlkjan6/scalr_geo_targets`; 0.5 m cells, next 8 m / 30 s of the same odometry segment, returns capped 2 m above the cell's lowest). Weight 10, Huber 0.05 m: about 0.1 against a distillation loss of 0.4. Label-free, dense (70,800 v2.4 and 52,945 v1.5 keyframes), and the same whichever day the patch was seen. |

Both: HELHEST_v2_4 corpus through `/mnt/personal/vlkjan6/setprop_v24`
(171,673 samples, held-out scenes excluded from the list), 26 epochs, 3 H200s,
`slurm/pretraining_h200_v10.batch` with `ARM=`. Submitted 2026-10-19 12:58: `v10_geo` job 11825912, `v10_centred` job 11825913, which crashed at its first log line (an initialisation displaced by the geo block, commit "distiller: prev_proprio back under the proprio block") and was resubmitted as 11826124 at 13:58. Smoke 11825788 (3,213 steps, 1.25 it/s on 1 H200): geo r2 rose to 0.39 / 0.33 / 0.43 for z-range / z-std / fraction (z_min 0.06), centred distillation 1.15, point diversity 0.11, teacher diversity 0.002 as centring predicts.
Not included: the session-adversarial head (C5) -- centring removes the reward
for the code; the GRL is the next guard if the probes below still find it.

**Gate while they run** (no GPU needed: `probe_backbone_value.py` on dgx07
CPU over the 1,119 probe frames, then `analyze_backbone_value.py --mlp` and
`analyze_backbone_why.py`): on `ckpt_last.pth` after ~1 day and at the end,
against random WI and v9 on the same cells,
(a) session identity from one block-36 cell near the hand-crafted 0.31 (v9 0.55),
(b) memorisation gap frames→sessions held out at or under 4 mm (v9 6.9),
(c) ≥ 80 % of block-36 variance within the frame (v9 71 %),
(d) the §3 MLP/ridge legs against random WI. Then one DRIFT arm on the passing
checkpoint, paired on v2.3 held-out against v24combo.

### 5.1 Epoch-6 gate (2026-10-21, checkpoints after 6 of 26 epochs)

CPU dump of both `ckpt_last.pth` snapshots (`/mnt/personal/vlkjan6/scratch_wi_probe/v10e6`,
same 1,119 frames and cells as the v24 probe, random WI and v9 re-dumped alongside;
`analysis_wedge60_mlp.txt`, `why.txt`). Session-held-out ridge on 35,266 label
cells, compression MAE with geom4 appended, mm:

| features | block 0 | 6 | 12 | 18 | 24 | 36 |
|---|---|---|---|---|---|---|
| random WI | 38.0 | 39.0 | 41.1 | 41.1 | 41.8 | 42.0 |
| v9 | | | | | | 41.6 |
| v10_centred | 35.9 | 35.7 | 33.7 | 33.6 | 35.2 | 37.5 |
| v10_geo | 37.2 | **32.6** | 33.0 | 34.3 | 34.1 | 38.4 |
| hand-crafted 16 | 39.0 | | | | | |

Paired per session against the hand-crafted stats: geo block 6 −4.8 ± 4.7 mm
(7/7 sessions better), centred block 12 −3.9 ± 4.2 (5/7). MLP leg (PCA-128,
256-256): geo block 12 41.3 mm against random block 12 49.8, hand-crafted
45.8, v9 block 36 136 (collapses). Frame-level readout: geo block 6 47.8 mm,
centred block 36 48.6, random 57.9, v9 53.0, hand-crafted 34.0.

Against the gate criteria: (c) within-frame variance share at block 36
centred 0.85, geo 0.80 (v9 0.71) PASS; (a) session identity from one block-36
cell centred 0.53, geo 0.52 (v9 0.55, random 0.46, hand-crafted 0.31) FAIL;
(b) memorisation gap centred +5.1 mm, geo +5.5 (v9 +6.9, random +4.4..+5.2,
target ≤ 4) marginal; (d) both arms beat random WI in every ridge and MLP leg
at blocks 6-24. Geometry content of a block-12 cell: 1 m z-std R² geo 0.94,
centred 0.92 (random 0.79, v9 0.82). Effective rank at block 36: centred 6.2,
geo 8.5 (v9 9.2, random 3.5).

Reading: the session code is reduced, not removed -- the last twelve blocks
still lose 4-6 mm against block 12 in both arms, geo head at block 36
notwithstanding. The mid-depth taps are the best cell features measured so
far. Verdict stays open until the end-of-training gate and one DRIFT arm
paired against v24_combo; if the end-of-training block 36 still trails block
12, the DRIFT arm should read the mid taps (or the GRL guard goes in).

### 5.2 Final gate (2026-10-24, both arms completed 2026-10-23, epoch 26 of 26)

Final `ckpt_last.pth` copied to `/mnt/personal/vlkjan6/scratch_wi_probe/v10final`
(v2.4 readout) and read again on HELHEST_v2_5 maps and labels
(`.../v10final_v25`). 1,104 of the 1,119 probe frames: DRIFT's loader gates
dropped 15, so the probe now resolves frames by cloud path. **Not like-for-like
with §5.1:** DRIFT now feeds v2 clouds with the Odin1 rig intensity constants
(120.8 / 46.3, DRIFT commit "Pass --intensity_norm explicitly; rig constants are
the default"), the convention v10 was pretrained on; the epoch-6 dump fed the v1
constants. Random WI is insensitive to this (42.0 -> 41.7 mm), v9 is not (41.6 -> 52.6).

Session-held-out ridge, compression MAE with geom4, mm (v2.4 / v2.5 target):

| features | block 0 | 6 | 12 | 18 | 36 |
|---|---|---|---|---|---|
| random WI | 37.1 / 35.2 | 39.8 / 37.0 | 41.1 / 38.3 | 41.0 / 38.5 | 41.7 / 39.0 |
| v9 | | | | | 52.6 / 45.9 |
| v10_centred | 39.1 / 37.1 | 36.7 / 35.7 | 36.1 / 35.1 | 46.9 / 44.7 | 48.7 / 45.3 |
| v10_geo | 36.9 / 36.3 | **33.5 / 32.6** | 35.1 / 34.2 | 37.9 / 36.8 | 45.9 / 42.8 |
| hand-crafted 16 | 39.2 / 38.5 | | | | |

Paired per session, geo block 6: −4.0 ± 4.9 mm against hand-crafted (6/7
sessions), −4.3 ± 5.7 against random WI block 0 on v2.4, −3.2 ± 4.4 on v2.5 --
inside one session sd. MLP leg (PCA-128): geo block 12 46.5 mm vs random 47.5 on
v2.4, 64.8 vs 44.1 on v2.5; block 36 of both arms collapses (110-134 mm). Block
36: within-frame variance 0.70 / 0.72 (v9 0.68), effective rank 24, session
identity from one cell 0.54-0.56 (hand-crafted 0.29), memorisation gap +7.7 mm.
v2.4 and v2.5 rank every row the same way; the v2.5 target is smaller (mean
0.095 m vs 0.169) and absolute MAE moves less than 3 mm.

Against the gate: (a) FAIL, (b) FAIL (best mid tap geo block 6 +5.3 mm, random
+3.9..+5.9), (c) FAIL at block 36, PASS at blocks 6-24 (0.84-0.89), (d) ridge
legs beat random WI at geo blocks 6-12 by 2-4 mm but within the session spread;
the MLP leg does not. Verdict: **neither final block-36 checkpoint replaces the
random backbone.** The session code returned in the last blocks as training ran
on (block 36 frame-within-session share 0.12-0.17 at epoch 6, 0.23-0.26 at the end,
partly confounded with the intensity change). The only candidate is a
truncated geo backbone read at block 6 (or 12), a 3-4 mm ridge gain the
session spread does not separate from zero; only a paired DRIFT arm against
v24_combo can settle it.

**2026-10-24 follow-up (user: "as expressive results as I can get").** (1) The
intensity confound is being closed with the two missing corners of
{epoch 6, final} x {v1_legacy, rig} on v2.4 (`scratch_wi_probe/v10e6_rig`,
`scratch_wi_probe/v10final_v1`). (2) Three DRIFT arms on the v26 recipe with the
backbone as the only variable, through DRIFT `slurm/v26_backbone.batch`: random
WI (11868090, amdgpulong), v10 geo cut at block 6 (`WI_v10_geo_L6.pth`,
11868088 + restart 11868089, amdgpu), v10 geo at 36 blocks (`WI_v10_geo.pth`,
11868091, amdgpuextralong). They are paired afterwards on the v2.5 held-out
scenes from their score dumps.

**Intensity factorial (done 2026-10-24 21:19).** Session-held-out ridge MAE
(+geom4, mm, v2.4, 1,104 frames), from `analysis_wedge60_mlp.txt` of
`v10e6`, `v10e6_rig`, `v10final_v1`, `v10final`:

| row | ep6 v1_legacy | ep6 rig | final v1_legacy | final rig |
|---|---|---|---|---|
| geo L36 | 38.4 | 44.0 | 42.1 | 45.9 |
| centred L36 | 37.5 | 49.7 | 40.1 | 48.7 |
| geo L6 | 32.6 | 33.8 | 34.7 | 33.5 |
| centred L12 | 33.7 | 36.1 | 34.0 | 36.1 |
| random L0 / L36 | 38.0 / 42.0 | 37.1 / 41.7 | 38.0 / 42.2 | 37.1 / 41.7 |
| v9 L36 | 41.6 | 52.6 | 41.1 | 52.6 |

At block 36 the intensity mode is the larger effect: rig costs geo +5.6 / +3.8
and centred +12.2 / +8.6 mm (epoch 6 / final), v9 +11. Further training at a
fixed mode costs geo +3.7 (v1) / +1.9 (rig) and centred +2.6 / -1.0. So about
two thirds of the epoch-6 -> final block-36 drop reported in §5.1/§5.2 is the
intensity change, not training. Under v1_legacy the final block 36 ties random
block 36 (geo paired +0.3 +- 4.0, centred +0.6 +- 6.6 mm) and still loses to
random block 0; the gate still fails at block 36 in every cell. The mid taps
barely move with intensity: geo L6 is the best v10 tap in every cell, paired
-3.9 to -5.3 mm against random, inside the session sd (3.4-5.7). Rig puts the
channel at mean 0.83 sd 0.41 (near what pretraining saw), v1_legacy at 7.30 sd
0.91, yet the deep blocks read better under v1_legacy; block-36 session
identity is also lower there (geo 0.48 vs 0.52, centred 0.51 vs 0.55 at final).
Unexplained; one reading is that rig lets the deep blocks use intensity as an
appearance code. Consequence for the DRIFT arms (INORM rig): geo6 is insensitive
to the mode; geo36 runs in its worst cell, and a v1_legacy geo36 sibling would
be the fair test of the full-depth backbone.
