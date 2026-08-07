from __future__ import annotations

import datetime
import re
from pathlib import Path

MARKER = "# === K345-I345 POLICY EMPTY-BATCH GUARD ==="


def apply_empty_batch_guard(path: str | Path) -> Path | None:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    compatible_markers = [
        MARKER,
        "# === OBJECT-POLICY EMPTY-BATCH GUARD V2 ===",
        "# === OBJECT-POLICY EMPTY-BATCH GUARD V1 ===",
        "# === REAL ADAPTIVE-KI EMPTY-BATCH GUARD ===",
        "# === ADAPTIVE-KI EMPTY-BATCH GUARD ===",
        "# === MIXED-K EMPTY-BATCH GUARD ===",
    ]
    if any(marker in text for marker in compatible_markers):
        return None

    pattern = re.compile(r"^(?P<indent>[ \t]*)def test_step\([^\n]*\):[ \t]*$", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"could not locate test_step() in {path}")
    indent = match.group("indent") + "    "
    guard = (
        "\n"
        f"{indent}{MARKER}\n"
        f"{indent}try:\n"
        f"{indent}    _policy_tco_init = batch.TCO_init\n"
        f"{indent}except AttributeError:\n"
        f"{indent}    print('[K345-I345] skip empty batch: no TCO_init', flush=True)\n"
        f"{indent}    return 0\n"
        f"{indent}try:\n"
        f"{indent}    _policy_pose_count = len(_policy_tco_init)\n"
        f"{indent}except TypeError:\n"
        f"{indent}    _policy_pose_count = None\n"
        f"{indent}if _policy_pose_count == 0:\n"
        f"{indent}    print('[K345-I345] skip empty batch: zero poses', flush=True)\n"
        f"{indent}    return 0\n"
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.before_empty_batch_{timestamp}.bak")
    backup.write_text(text, encoding="utf-8")
    path.write_text(text[: match.end()] + guard + text[match.end() :], encoding="utf-8")
    return backup
