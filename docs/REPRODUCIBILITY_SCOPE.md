# Reproducibility scope and limitations

## What the clean code reproduces directly

- LM-O target and hypothesis coverage checks.
- Physical score-based Top-K pruning.
- The frozen object-specific K/I policy.
- Group construction and expected row counts.
- MegaPose refinement command construction using the upstream GigaPose repository.
- Merge validation and prevention of duplicate/missing targets.
- Reconstructed grouped compute time.
- Official BOP19 evaluation invocation.
- SHA256 manifests for inputs and outputs.

## What remains an upstream/external dependency

- GigaPose and MegaPose source code.
- Model checkpoints.
- BOP data, targets, models and rendered templates.
- CNOS-FastSAM detections and CSM-filtered detections.
- The accelerated coarse model patch implementing TorchAO INT8, FP16 autocast and IST caching.

The exact historical combined-coarse script is included in `legacy_reference/`. It uses the original experiment workspace and restores the corresponding source artifacts. The public clean interface deliberately avoids silently downloading multi-gigabyte models or altering an upstream repository without an explicit user action.

## Timing caveat

The Top-3 result uses a directly recorded time field. The Object-adaptive result uses a reconstructed measured time. The two values should be compared descriptively and with their timing construction stated.
