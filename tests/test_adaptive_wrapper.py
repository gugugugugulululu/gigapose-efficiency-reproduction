from __future__ import annotations

from pathlib import Path


def test_adaptive_group_tsv_is_not_built_from_python_stdout() -> None:
    repo = Path(__file__).resolve().parents[1]

    script = (
        repo
        / "scripts"
        / "run_lmo_adaptive.sh"
    )

    text = script.read_text(
        encoding="utf-8"
    )

    # The historical bug redirected Python stdout directly into the TSV.
    # The formal runtime can print an EGL bootstrap message to stdout,
    # which then becomes an invalid first group row.
    vulnerable = (
        '"$PYTHON_BIN" - "$PREPARED_DIR/manifest.json" '
        '> "$GROUP_TSV"'
    )

    assert vulnerable not in text

    # The fixed wrapper passes the output path as argv[2] and has Python
    # write directly to that file.
    expected = (
        '"$PYTHON_BIN" - "$PREPARED_DIR/manifest.json" '
        '"$GROUP_TSV"'
    )

    assert expected in text

    assert (
        'output_path = Path(sys.argv[2])'
        in text
    )

    assert (
        'group_commands.tsv rows='
        in text
    )

    assert (
        'expected=5'
        in text
    )
