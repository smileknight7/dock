#!/usr/bin/env python3
"""Shared helpers for Vina docking box and run_vina."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def write_vina_box_files(
    *,
    json_path: Path,
    txt_path: Path,
    receptor_pdbqt: Path,
    box_center: Sequence[float],
    box_size: Sequence[float],
    metadata: dict[str, object],
) -> None:
    json_path.write_text(
        json.dumps({"receptor_pdbqt": str(receptor_pdbqt), "center": box_center,
                    "size": box_size, **metadata}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"receptor = {receptor_pdbqt}",
        f"center_x = {box_center[0]:.3f}", f"center_y = {box_center[1]:.3f}", f"center_z = {box_center[2]:.3f}",
        f"size_x = {box_size[0]:.3f}", f"size_y = {box_size[1]:.3f}", f"size_z = {box_size[2]:.3f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
