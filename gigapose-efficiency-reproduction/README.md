# GigaPose–MegaPose Inference-Efficiency Reproduction

Clean reproduction code for the two LM-O configurations used in the dissertation:

1. **Combined Top-3**: accelerated GigaPose coarse predictions, physical Top-3 pose-hypothesis pruning, and five MegaPose refinement iterations.
2. **Combined Object-adaptive**: the same accelerated coarse source with a frozen object-specific `K=3–5 / I=3–5` policy, grouped MegaPose refinement, merged predictions, reconstructed measured compute time, and official BOP19 evaluation.

The repository does not redistribute GigaPose, MegaPose, BOP datasets, model weights, or rendered templates. Those assets must be obtained from their original sources.

## Reproduction scope

The clean entry points start from a **frozen accelerated coarse result** containing:

- one 1,445-row LM-O main BOP CSV;
- one 7,225-row LM-O `MultiHypothesis` CSV;
- five hypotheses for each of the 1,445 targets;
- all 200 LM-O test images and object IDs `1, 5, 6, 8, 9, 10, 11, 12`.

This boundary makes the refinement policies directly inspectable and repeatable. The exact historical Colab scripts that generated the combined coarse source and ran the original full experiments are preserved under `legacy_reference/` for provenance.

## Fixed reference assets

| Item | Reference |
|---|---|
| GigaPose commit | `388e8bddd8a5443e284a7f70ad103d03f3f461c5` |
| GigaPose checkpoint SHA256 | `0f60a23b03ddc41d2135c916ed1e66fb16f814f612dbde0305ae5a2c0f45c932` |
| Python used in formal runs | 3.12.12 |
| PyTorch used in formal runs | 2.5.1+cu121 |
| GPU used in formal runs | Tesla T4 |
| LM-O images | 200 |
| LM-O targets | 1,445 |
| Original hypotheses | 7,225 |
| Top-3 hypotheses | 4,335 |
| Adaptive hypotheses | 5,268 |

Runtime values are hardware- and timing-boundary-specific. Accuracy and row-coverage checks are expected to be more stable than wall-clock measurements.

## Repository layout

```text
configs/lmo/                 Frozen policy and expected asset metadata
scripts/                     Clean runnable entry points
src/gigapose_efficiency/     CSV, policy, merge, timing and validation code
tests/                       CPU-only unit tests
expected/                    Dissertation reference results
patches/                     Compatibility-patch documentation
legacy_reference/            Exact historical Colab shell cells
```

## Installation

```bash
git clone <your-repository-url>
cd gigapose-efficiency-reproduction
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

For the GPU experiments, use the existing GigaPose/MegaPose environment rather than the small CPU-only virtual environment above. Add this repository to `PYTHONPATH`, or install it with that environment's Python:

```bash
/content/gp_exec/bin/python -m pip install -e .
```

## Required workspace layout

The default scripts assume a workspace similar to:

```text
bop_workspace/
├── code/
│   ├── gigapose/
│   └── bop_toolkit/
├── datasets/
│   └── lmo/
├── pretrained/
│   ├── gigaPose_v1.ckpt
│   └── megapose-models/
└── results/
```

All paths can be passed explicitly. No Google Drive path is hard-coded in the clean scripts.

## Check assets

```bash
bash scripts/check_assets.sh \
  --workspace /path/to/bop_workspace \
  --python /content/gp_exec/bin/python
```

## Combined Top-3

```bash
bash scripts/run_lmo_top3.sh \
  --workspace /path/to/bop_workspace \
  --python /content/gp_exec/bin/python \
  --env-script /path/to/env.sh \
  --coarse-main /path/to/accelerated_coarse.csv \
  --coarse-multi /path/to/accelerated_coarse_MultiHypothesis.csv
```

The script performs the following operations:

```text
7,225 coarse hypotheses
→ stable score sorting within each target
→ physical Top-3 retention
→ 4,335 hypotheses
→ MegaPose K=3, I=5, batch 8×8
→ 1,445 final predictions
→ official BOP19 evaluation
```

The physical CSV truncation is deliberate: changing only a Hydra value is not accepted as evidence that the refinement input actually contained three hypotheses per target.

## Combined Object-adaptive

```bash
bash scripts/run_lmo_adaptive.sh \
  --workspace /path/to/bop_workspace \
  --python /content/gp_exec/bin/python \
  --env-script /path/to/env.sh \
  --coarse-main /path/to/accelerated_coarse.csv \
  --coarse-multi /path/to/accelerated_coarse_MultiHypothesis.csv
```

Frozen policy:

| Object | ID | K | I |
|---|---:|---:|---:|
| ape | 1 | 4 | 4 |
| can | 5 | 5 | 4 |
| cat | 6 | 3 | 4 |
| driller | 8 | 3 | 5 |
| duck | 9 | 4 | 4 |
| eggbox | 10 | 4 | 4 |
| glue | 11 | 3 | 5 |
| holepuncher | 12 | 3 | 3 |

Execution groups:

```text
K=3, I=3 → object 12
K=3, I=4 → object 6
K=3, I=5 → objects 8, 11
K=4, I=4 → objects 1, 9, 10
K=5, I=4 → object 5
```

The script physically builds five group inputs, runs five fresh refinement jobs, merges exactly 1,445 final predictions, reconstructs per-image compute time, assigns the reconstructed per-image time consistently to the merged BOP CSV, and runs official BOP19 evaluation.

### Important interpretation

The policy was selected using object-level results from the same LM-O test workload. It is therefore an **offline test-set oracle ablation**, not an online adaptive policy. Its compute time is reconstructed from measured group contributions rather than obtained from one integrated single-process wall-clock run.

## Runtime reconstruction

For the default `group_time_mode=total`, the clean code uses:

```text
full accelerated-coarse time counted once
+ Σ(refined group time − source group coarse time)
```

The group increments are added only on images where that policy group is active, and the final mean is calculated over the complete 200-image LM-O split.

## Outputs

Each run writes:

```text
inputs or prepared groups
logs/refine.log
predictions/<run>_lmo-test.csv
reports/*_validation.json
reports/sha256_manifest.json
reports/reconstructed_runtime.json   # adaptive only
bop19/eval/                          # official evaluation
```

A live terminal monitor is available:

```bash
bash scripts/monitor.sh 30 /path/to/refine.log
```

## Expected dissertation values

The values in `expected/lmo_expected_results.json` are comparison references, not hard pass/fail thresholds for a different GPU session.

| Configuration | AR | Mean pipeline compute time |
|---|---:|---:|
| Combined Top-3 | 59.14% | 4.48 s/image, directly recorded |
| Combined Object-adaptive | 59.54% | 4.73 s/image, reconstructed |

## Historical exact scripts

`legacy_reference/` contains the exact long-form shell cells recovered from the implementation records:

- `run_lmo_combined_coarse_historical.sh`
- `run_lmo_combined_top3_historical.sh`
- `resume_lmo_top3_bop19_historical.sh`
- `run_lmo_object_adaptive_historical.sh`

They preserve the original absolute paths, environment restoration, compatibility patches and run IDs. They are included for auditability, but the clean scripts are the recommended public interface.

## References

- [GigaPose](https://github.com/nv-nguyen/gigapose)
- [GigaPose paper](https://openaccess.thecvf.com/content/CVPR2024/html/Nguyen_GigaPose_Fast_and_Robust_Novel_Object_Pose_Estimation_via_One_CVPR_2024_paper.html)
- [MegaPose](https://github.com/megapose6d/megapose6d)
- [MegaPose paper](https://arxiv.org/abs/2212.06870)
- [BOP Benchmark](https://bop.felk.cvut.cz/)
- [BOP Toolkit](https://github.com/thodan/bop_toolkit)
