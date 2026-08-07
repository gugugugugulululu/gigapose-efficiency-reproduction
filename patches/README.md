# Compatibility patches

The clean adaptive runner can apply the small MegaPose empty-batch guard automatically:

```bash
gigapose-efficiency patch-empty-batch \
  --refiner /path/to/gigapose/src/models/refiner.py
```

The patch creates a timestamped backup and inserts a guard at the start of `test_step`. It skips images with no initial poses for the active object group. This is needed because an object-policy group may cover fewer than all 200 LM-O images.

The historical scripts also contain the Python 3.12 dataclass and NumPy/Panda3D compatibility repairs used in the original environment. Those larger environment-specific repairs are retained in `legacy_reference/` rather than applied automatically by the clean package.
