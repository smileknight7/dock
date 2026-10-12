#!/usr/bin/env python3
"""Shared configuration and helpers for AutoDock4 scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path


MGLTOOLS_ROOT = Path("/data3/lpw/software/autodock4/mgltools_x86_64Linux2_1.5.7")
PYTHONSH = MGLTOOLS_ROOT / "bin" / "pythonsh"
PREPARE_LIGAND4 = MGLTOOLS_ROOT / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24" / "prepare_ligand4.py"          # 脚本运行路径
PREPARE_RECEPTOR4 = MGLTOOLS_ROOT / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24" / "prepare_receptor4.py"
PREPARE_GPF4 = MGLTOOLS_ROOT / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24" / "prepare_gpf4.py"
PREPARE_DPF4 = MGLTOOLS_ROOT / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24" / "prepare_dpf4.py"

AUTOGRID4 = Path("/data3/lpw/software/autodock4/x86_64Linux2/autogrid4")
AUTODOCK4 = Path("/data3/lpw/software/autodock4/x86_64Linux2/autodock4")

GRID_SPACING = 0.375


def run_command(cmd: list[str], cwd: Path, label: str = "command") -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )

    if result.returncode:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"{label} failed:\n{msg}")


def run_mgltools_script(script: Path, args: list[str], cwd: Path) -> None:
    run_command([PYTHONSH, script, *args], cwd, script.name)


def run_autogrid4(gpf: Path, glg: Path) -> None:
    run_command([AUTOGRID4, "-p", gpf, "-l", glg], gpf.parent, "autogrid4")


def run_autodock4(dpf: Path, dlg: Path) -> None:
    run_command([AUTODOCK4, "-p", dpf, "-l", dlg], dpf.parent, "autodock4")