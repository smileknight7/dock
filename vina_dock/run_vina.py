#!/usr/bin/env python3
"""Dock prepared ligand PDBQT files with AutoDock Vina.

Examples:
    python vina/run_vina.py \
        --receptor-pdbqt /path/to/receptor.pdbqt \
        --box-config /path/to/receptor.box.json \
        --ligand-pdbqt /path/to/ligand.pdbqt \
        --output-dir /path/to/work/docking

    python vina/run_vina.py \
        --receptor-pdbqt /data3/lpw/dock_workdir/vina/receptor/receptor.pdbqt \
        --box-config /data3/lpw/dock_workdir/vina/receptor/receptor.box.json \
        --ligand-dir /data3/lpw/dock_workdir/vina/ligands \
        --output-dir /data3/lpw/dock_workdir/vina/dock_result \
        --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from vina import Vina

from .vina_utils import (
    LigandSpec,
    ensure_directory,
    ligand_spec_from_path,
    parse_triplet,
    positive_int,
    read_json,
    write_json,
)


# ---------------------------------------------------------------------------
# Box resolution
# ---------------------------------------------------------------------------

def resolve_box(
    box_config: Path | None,
    box_center: tuple[float, float, float] | None,
    box_size: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if box_config is not None:
        if box_center is not None or box_size is not None:
            raise ValueError("Use either --box-config or --box-center/--box-size")
        cfg = read_json(box_config.resolve())
        return tuple(cfg["center"]), tuple(cfg["size"])
    if box_center is None or box_size is None:
        raise ValueError("Provide --box-config or both --box-center and --box-size")
    return box_center, box_size


# ---------------------------------------------------------------------------
# Ligand resolution
# ---------------------------------------------------------------------------

def resolve_specs(args: argparse.Namespace) -> list[LigandSpec]:
    if args.ligand_pdbqt is not None:
        spec = ligand_spec_from_path(args.ligand_pdbqt)
        if spec.source_kind != "pdbqt":
            raise ValueError("--ligand-pdbqt must be a .pdbqt file")
        return [spec]
    pattern = args.ligand_dir.rglob("*.pdbqt") if not args.no_recursive else args.ligand_dir.glob("*.pdbqt")
    specs = [ligand_spec_from_path(p) for p in sorted(pattern)]
    if not specs:
        raise ValueError(f"No .pdbqt files under {args.ligand_dir}")
    return specs


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------

def dock_one(
    spec: LigandSpec, *,
    output_dir: Path, receptor_pdbqt: Path,
    box_center: tuple[float, float, float], box_size: tuple[float, float, float],
    score_function: str, cpu: int, seed: int, no_refine: bool,
    exhaustiveness: int, n_poses: int, min_rmsd: float, energy_range: float, max_evals: int,
    verbose: bool, skip_existing: bool,
) -> dict[str, object]:
    work_dir = ensure_directory(output_dir / spec.ligand_id)
    poses_pdbqt = work_dir / f"{spec.ligand_id}_out.pdbqt"
    summary_json = work_dir / "summary.json"

    if skip_existing and summary_json.exists() and poses_pdbqt.exists():
        payload = read_json(summary_json)
        payload["status"] = "skipped"
        return payload

    if spec.path is None or spec.source_kind != "pdbqt":
        raise ValueError("run_vina.py only accepts prepared .pdbqt inputs")

    v = Vina(sf_name=score_function, cpu=cpu, seed=seed, no_refine=no_refine, verbosity=1 if verbose else 0)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_file(str(spec.path.resolve()))
    v.compute_vina_maps(center=box_center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses, min_rmsd=min_rmsd, max_evals=max_evals)
    v.write_poses(str(poses_pdbqt), n_poses=n_poses, energy_range=energy_range, overwrite=True)
    energies = v.energies(n_poses=n_poses, energy_range=energy_range)

    summary = {
        "ligand_id": spec.ligand_id, "status": "ok",
        "source": spec.source, "ligand_pdbqt": str(spec.path),
        "poses_pdbqt": str(poses_pdbqt), "receptor_pdbqt": str(receptor_pdbqt),
        "score_function": score_function,
        "box_center": list(box_center), "box_size": list(box_size),
        "best_score": float(energies[0][0]), "pose_count": len(energies),
        "energies": [[float(v) for v in row] for row in energies.tolist()],
    }
    write_json(summary_json, summary)
    return summary


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

_CSV_FIELDS = ["ligand_id", "status", "best_score", "pose_count", "poses_pdbqt", "ligand_pdbqt", "source", "error"]


def write_results_csv(results: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, "") for k in _CSV_FIELDS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    receptor_pdbqt = args.receptor_pdbqt.resolve()
    output_dir = ensure_directory(args.output_dir.resolve())
    center, size = resolve_box(args.box_config, parse_triplet(args.box_center, "--box-center"), parse_triplet(args.box_size, "--box-size"))
    specs = resolve_specs(args)
    jobs = max(1, min(args.jobs, len(specs)))

    task_kw = dict(
        output_dir=output_dir, receptor_pdbqt=receptor_pdbqt,
        box_center=center, box_size=size,
        score_function=args.score_function, cpu=args.cpu, seed=args.seed, no_refine=args.no_refine,
        exhaustiveness=args.exhaustiveness, n_poses=args.n_poses, min_rmsd=args.min_rmsd,
        energy_range=args.energy_range, max_evals=args.max_evals,
        verbose=args.verbose, skip_existing=args.skip_existing,
    )

    results: list[dict[str, object]] = []
    show_progress = len(specs) > 1 and sys.stderr.isatty()

    def _record(spec: LigandSpec, outcome: dict | None = None, error: Exception | None = None) -> dict[str, int]:
        if outcome is not None:
            results.append(outcome)
        else:
            results.append({"ligand_id": spec.ligand_id, "status": "failed", "source": spec.source, "error": str(error)})
        counts = {"ok": 0, "skipped": 0, "failed": 0}
        for r in results:
            s = str(r.get("status", "failed"))
            if s in counts:
                counts[s] += 1
        return counts

    if jobs == 1:
        pbar = tqdm(specs, desc="Docking", unit="lig") if show_progress else specs
        for spec in pbar:
            try:
                c = _record(spec, outcome=dock_one(spec, **task_kw))
            except Exception as exc:
                c = _record(spec, error=exc)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(**c)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(dock_one, s, **task_kw): s for s in specs}
            pbar = tqdm(total=len(specs), desc="Docking", unit="lig") if show_progress else None
            for fut in as_completed(futs):
                spec = futs[fut]
                try:
                    c = _record(spec, outcome=fut.result())
                except Exception as exc:
                    c = _record(spec, error=exc)
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(**c)
            if pbar:
                pbar.close()

    results.sort(key=lambda r: str(r["ligand_id"]))
    write_results_csv(results, output_dir / "results.csv")

    for row in results:
        if row["status"] in ("ok", "skipped"):
            print(f"{row['ligand_id']}\t{row['status']}\t{row.get('best_score', '')}\t{row.get('poses_pdbqt', '')}")
        else:
            print(f"FAIL\t{row['ligand_id']}\t{row.get('error', '')}", file=sys.stderr)

    return 0 if all(r["status"] in ("ok", "skipped") for r in results) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receptor-pdbqt", type=Path, default='/data3/lpw/dock_workdir/vina/receptor/3I03_protein.pdbqt', help="Prepared receptor PDBQT from prepare_receptor.py.")
    p.add_argument("--box-config", type=Path, default='/data3/lpw/dock_workdir/vina/receptor/3I03_protein.box.json', help="Box JSON from prepare_receptor.py.")
    p.add_argument("--box-center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--box-size", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))

    p.add_argument("--ligand-pdbqt", type=Path, help="Single ligand PDBQT.")
    p.add_argument("--ligand-dir", type=Path, default= '/data3/lpw/dock_workdir/vina/ligands', help="Directory of prepared .pdbqt files.")

    p.add_argument("--output-dir", type=Path, default='/data3/lpw/dock_workdir/vina/dock_result', help="Output directory for docking results.")
    p.add_argument("--score-function", choices=("vina", "vinardo", "ad4"), default="vina")
    p.add_argument("--cpu", type=int, default=8)                                    # cpu数量和exhaustiveness 要相称，有1个cpu就走一次MC计算
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exhaustiveness", type=positive_int, default=8)
    p.add_argument("--n-poses", type=positive_int, default=9)
    p.add_argument("--min-rmsd", type=float, default=1.0)
    p.add_argument("--energy-range", type=float, default=3.0)
    p.add_argument("--max-evals", type=int, default=0)
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--jobs", type=positive_int, default=1, help="Parallel docking workers.")
    try:
        raise SystemExit(main(p.parse_args()))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
