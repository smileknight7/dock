#!/usr/bin/env python3
"""Shared helpers for Vina receptor preparation, docking box, and run_vina."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem


SUPPORTED_DOCKING_SUFFIXES = {".sdf", ".mol2", ".mol", ".pdb", ".pdbqt"}
PDBQT_ATOM_RECORDS = ("ATOM  ", "HETATM")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LigandSpec:
    ligand_id: str
    source_kind: str  # "file" | "pdbqt" | "smiles"
    source: str
    path: Path | None = None
    smiles: str | None = None


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise ValueError("must be >= 1")
    return n


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_identifier(value: str | None, fallback: str) -> str:
    raw = (value or "").strip() or fallback
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return sanitized or fallback


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_triplet(values: list[float] | None, flag_name: str) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError(f"{flag_name} expects exactly 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


# ---------------------------------------------------------------------------
# Molecule I/O  (RDKit)
# ---------------------------------------------------------------------------

def load_single_molecule(path: Path) -> Chem.Mol:
    suffix = path.suffix.lower()
    loaders = {
        ".sdf": lambda: next((m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None), None),
        ".mol": lambda: Chem.MolFromMolFile(str(path), removeHs=False),
        ".mol2": lambda: Chem.MolFromMol2File(str(path), removeHs=False),
        ".pdb": lambda: Chem.MolFromPDBFile(str(path), removeHs=False),
    }
    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported format: {path}")
    mol = loader()
    if mol is None:
        raise ValueError(f"RDKit failed to read {path}")
    return mol


# ---------------------------------------------------------------------------
# Receptor preparation  (mk_prepare_receptor.py CLI)
# ---------------------------------------------------------------------------

def prepare_receptor_pdbqt(
    *,
    receptor_file: Path,
    output_pdbqt: Path,
    charge_model: str | None = None,
    allow_bad_res: bool = False,
) -> None:
    """Prepare receptor PDBQT via mk_prepare_receptor.py CLI.

    Requires `mk_prepare_receptor.py` on PATH (ships with Meeko).
    HIS ambiguity, residue templates, and atom typing are handled by the CLI.
    """
    output_pdbqt = Path(output_pdbqt)
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    basename = output_pdbqt.with_suffix("")

    cmd = [
        "mk_prepare_receptor.py",
        "-i", str(receptor_file),
        "-o", str(basename),
        "-p",
    ]
    if charge_model:
        cmd.extend(["--charge_model", charge_model])
    if allow_bad_res:
        cmd.append("--allow_bad_res")

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mk_prepare_receptor.py failed: {r.stderr or r.stdout}")

    # CLI may produce <basename>.pdbqt or <basename>_rigid.pdbqt depending on version
    if not output_pdbqt.exists():
        candidates = sorted(basename.parent.glob(f"{basename.name}*.pdbqt"))
        if not candidates:
            raise FileNotFoundError(f"No PDBQT produced for {basename}")
        candidates[0].replace(output_pdbqt)


# ---------------------------------------------------------------------------
# Docking box
# ---------------------------------------------------------------------------

def coordinates_from_pdbqt(path: Path) -> np.ndarray:
    coords = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(PDBQT_ATOM_RECORDS):
            continue
        try:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            parts = line.split()
            if len(parts) >= 8:
                coords.append([float(parts[5]), float(parts[6]), float(parts[7])])
    if not coords:
        raise ValueError(f"No atom coordinates in {path}")
    return np.asarray(coords, dtype=float)


def load_reference_coordinates(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".pdbqt":
        return coordinates_from_pdbqt(path)
    mol = Chem.AddHs(load_single_molecule(path), addCoords=True)
    if mol.GetNumConformers() == 0:
        raise ValueError(f"No conformer in {path}")
    conf = mol.GetConformer()
    return np.array([[p.x, p.y, p.z] for p in (conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms()))])


def compute_box_from_reference(
    reference_path: Path, padding: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    coords = load_reference_coordinates(reference_path)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    center = tuple(float(v) for v in (lo + hi) / 2)
    size = tuple(float(v) for v in (hi - lo) + 2 * padding)
    return center, size


def validate_box_args(
    *,
    reference_ligand: Path | None,
    box_center: tuple[float, float, float] | None,
    box_size: tuple[float, float, float] | None,
) -> None:
    has_ref = reference_ligand is not None
    has_manual = box_center is not None or box_size is not None
    if has_ref == has_manual:
        raise ValueError("Use either --reference-ligand or --box-center with --box-size")
    if (box_center is None) != (box_size is None):
        raise ValueError("--box-center and --box-size must be provided together")


def write_vina_box_files(
    *,
    json_path: Path,
    txt_path: Path,
    receptor_pdbqt: Path,
    box_center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    metadata: dict[str, object],
) -> None:
    write_json(json_path, {"receptor_pdbqt": str(receptor_pdbqt), "center": list(box_center), "size": list(box_size), **metadata})
    lines = [
        f"receptor = {receptor_pdbqt}",
        f"center_x = {box_center[0]:.3f}", f"center_y = {box_center[1]:.3f}", f"center_z = {box_center[2]:.3f}",
        f"size_x = {box_size[0]:.3f}", f"size_y = {box_size[1]:.3f}", f"size_z = {box_size[2]:.3f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Ligand spec (used by run_vina.py for resolving --ligand input)
# ---------------------------------------------------------------------------

def ligand_spec_from_path(path: Path, ligand_id: str | None = None) -> LigandSpec:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DOCKING_SUFFIXES:
        raise ValueError(f"Unsupported ligand file: {path}")
    return LigandSpec(
        ligand_id=sanitize_identifier(ligand_id or path.stem, "ligand"),
        source_kind="pdbqt" if suffix == ".pdbqt" else "file",
        source=str(path), path=path.resolve(),
    )
