from __future__ import annotations

from pathlib import Path


def test_bop_wrapper_pins_python_and_isolates_mesa_renderer() -> None:

    root = Path(__file__).resolve().parents[1]

    script = (
        root
        / "scripts"
        / "evaluate_bop19.sh"
    )

    text = script.read_text(
        encoding="utf-8"
    )

    # Nested BOP scripts call bare "python".
    assert (
        'export PATH="$PYTHON_DIR:$PATH"'
        in text
    )

    assert (
        'RESOLVED_CHILD_PYTHON="$('
        in text
    )

    # Evaluation must not inherit the GigaPose
    # Panda3D/NVIDIA EGL renderer state.
    assert (
        "unset GIGAPOSE_FORCE_NVIDIA_EGL"
        in text
    )

    assert (
        "export PYOPENGL_PLATFORM=egl"
        in text
    )

    assert (
        "export EGL_PLATFORM=surfaceless"
        in text
    )

    assert (
        "export VISPY_APP_BACKEND=egl"
        in text
    )

    assert (
        "export LIBGL_ALWAYS_SOFTWARE=1"
        in text
    )

    assert (
        "export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe"
        in text
    )

    assert (
        "export GALLIUM_DRIVER=llvmpipe"
        in text
    )

    # Single-worker is the stable default used by
    # the validated Mesa-surfaceless evaluation.
    assert (
        'NUM_WORKERS="${BOP_EVAL_NUM_WORKERS:-1}"'
        in text
    )

    assert (
        '--num_workers="$NUM_WORKERS"'
        in text
    )

    # Explicit data root.
    assert (
        'export BOP_PATH="$WORKSPACE/datasets"'
        in text
    )

    # Official evaluator is still used.
    assert (
        'scripts/eval_bop19_pose.py'
        in text
    )

    assert (
        "--renderer_type=vispy"
        in text
    )
