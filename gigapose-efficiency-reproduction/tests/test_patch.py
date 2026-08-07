from __future__ import annotations

from pathlib import Path

from gigapose_efficiency.patches import MARKER, apply_empty_batch_guard


def test_empty_batch_patch_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "refiner.py"
    path.write_text(
        "class Refiner:\n"
        "    def test_step(self, batch, batch_idx):\n"
        "        return batch.TCO_init\n",
        encoding="utf-8",
    )
    backup = apply_empty_batch_guard(path)
    assert backup is not None and backup.exists()
    assert MARKER in path.read_text(encoding="utf-8")
    assert apply_empty_batch_guard(path) is None
