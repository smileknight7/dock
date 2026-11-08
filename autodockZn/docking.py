#!/usr/bin/env python3
"""Dock ligands against a shared AutoDock4Zn grid with the Vina Python API.

The --maps argument is the map filename prefix, such as
/path/to/proteinH_zn for /path/to/proteinH_zn.C.map. It is not a GPF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import vina
import multiprocessing

CSV_FIELDS = ("ligand", "status", "best_affinity", "poses_pdbqt", "log", "error")


def resolve_maps_prefix(value: Path) -> Path:
    name = str(value.expanduser())
    if name.endswith(".maps.fld"):
        name = name[: -len(".maps.fld")]
    elif name.endswith(".gpf"):
        raise ValueError("--maps needs the map prefix, not a .gpf file")

    prefix = Path(name).resolve()
    for suffix in (".maps.fld", ".e.map", ".d.map"):
        path = Path(f"{prefix}{suffix}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing grid file: {path}")
    return prefix


def ligand_atom_types(ligand: Path) -> set[str]:
    types = {
        line.split()[-1]
        for line in ligand.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    }
    if not types:
        raise ValueError(f"No ATOM/HETATM records in ligand PDBQT: {ligand}")
    return types




def best_affinity(poses_pdbqt: Path) -> float | None:
    with poses_pdbqt.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if parts[:3] == ["REMARK", "VINA", "RESULT:"]:
                return float(parts[3])
    return None


def dock_one(
    ligand: Path,
    poses_pdbqt: Path,
    maps_prefix: Path,
    exhaustiveness: int,
    cpu: int,
    num_modes: int,
    energy_range: float,
    seed: int | None,
    skip_existing: bool,
) -> dict[str, object]:
    log_path = poses_pdbqt.with_suffix(".log")
    row: dict[str, object] = {
        "ligand": str(ligand),
        "status": "failed",
        "best_affinity": "",
        "poses_pdbqt": str(poses_pdbqt),
        "log": str(log_path),
        "error": "",
    }
    try:

        poses_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and poses_pdbqt.is_file() and poses_pdbqt.stat().st_size:
            row.update(status="skipped", best_affinity=best_affinity(poses_pdbqt))
            return row
        v = vina.Vina(sf_name="ad4", cpu=cpu, seed=seed or 0, verbosity=0)
        v.load_maps(str(maps_prefix))
        v.set_ligand_from_file(str(ligand))
        v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
        v.write_poses(str(poses_pdbqt), n_poses=num_modes,
                      energy_range=energy_range, overwrite=True)
        if not poses_pdbqt.is_file() or not poses_pdbqt.stat().st_size:
            raise RuntimeError("Vina produced no poses")
        score = best_affinity(poses_pdbqt)
        if score is None:
            raise RuntimeError(f"No VINA RESULT record in {poses_pdbqt}")
        row.update(status="ok", best_affinity=score)
        log_path.write_text(
            f"Vina Python API; scoring=ad4\nligand={ligand}\nmaps={maps_prefix}\n"
            f"exhaustiveness={exhaustiveness}\ncpu={cpu}\n"
            f"num_modes={num_modes}\nenergy_range={energy_range}\n"
            f"seed={seed if seed is not None else 0}\nbest_affinity={score}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        row["error"] = str(exc)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"Vina Python API failed: {exc}\n", encoding="utf-8")
    return row


def build_jobs(args) -> list[tuple[Path, Path]]:
    output_dir = args.output_dir.resolve()
    if args.ligand is not None:
        ligand = args.ligand.resolve()
        return [(ligand, output_dir / f"{ligand.stem}_ad4_out.pdbqt")]

    if args.ligand_dir is None:
        raise ValueError("Provide --ligand or --ligand-dir")
    ligand_dir = args.ligand_dir.resolve()
    paths = ligand_dir.rglob("*.pdbqt") if args.recursive else ligand_dir.glob("*.pdbqt")
    ligands = sorted(path.resolve() for path in paths)
    if not ligands:
        raise FileNotFoundError(f"No ligand PDBQT files in {ligand_dir}")
    return [
            (
                ligand,
                output_dir / ligand.parent.relative_to(ligand_dir) / f"{ligand.stem}_ad4_out.pdbqt",
            )
            for ligand in ligands
    ]


def run_job(job: tuple) -> dict[str, object]:
    """Unpack one job for multiprocessing.Pool.imap_unordered."""
    return dock_one(*job)


def run(args: argparse.Namespace) -> int:
    maps_prefix = resolve_maps_prefix(args.maps)
    jobs = build_jobs(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    options = (
        maps_prefix, args.exhaustiveness, args.cpu,
        args.num_modes, args.energy_range, args.seed, args.skip_existing,
    )
    job_args = [(ligand, output, *options) for ligand, output in jobs]
    rows: list[dict[str, object]] = []
    with multiprocessing.Pool(processes=min(args.jobs, len(jobs))) as pool:
        for row in tqdm(pool.imap_unordered(run_job, job_args),
                        total=len(jobs), desc="Vina AD4 docking"):
            rows.append(row)
            if row["status"] == "failed":
                print(f"FAIL\t{row['ligand']}\t{row['error']}", file=sys.stderr)
            else:
                print(f"{str(row['status']).upper()}\t{row['ligand']}\t{row['best_affinity']}")

    rows.sort(key=lambda row: str(row["ligand"]))
    return 1 if any(row["status"] == "failed" for row in rows) else 0





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ligand", type=Path, help="One prepared ligand PDBQT.")
    parser.add_argument("--ligand-dir", type=Path, help="Directory of ligand PDBQT files.")

    parser.add_argument("--maps", type=Path, required=True, help="Grid map prefix, e.g. /path/proteinH_zn.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--cpu", type=int, default=1, help="CPUs per ligand.")
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--energy-range", type=float, default=3.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--jobs", "--num_workers", type=int, default=1,
                        help="Concurrent ligands; total CPU use is approximately jobs × cpu.")
    parser.add_argument("--recursive", action="store_true", help="Find ligands in subdirectories.")
    parser.add_argument("--skip-existing", action="store_true")
    raise SystemExit(run(parser.parse_args()))
