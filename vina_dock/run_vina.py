#!/usr/bin/env python3
"""Dock prepared ligand PDBQT files with AutoDock Vina.

Examples:
    python -m vina_dock.run_vina \
        --receptor-pdbqt /path/to/receptor.pdbqt \
        --box-config /path/to/receptor.box.json \
        --ligand-pdbqt /path/to/ligand.pdbqt \
        --output-dir /path/to/work/docking

    python -m vina_dock.run_vina \
        --receptor-pdbqt /data3/lpw/dock_workdir/vina/receptor/receptor.pdbqt \
        --box-config /data3/lpw/dock_workdir/vina/receptor/receptor.box.json \
        --ligand-dir /data3/lpw/dock_workdir/vina/ligands \
        --output-dir /data3/lpw/dock_workdir/vina/dock_result \
        --jobs 4
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from vina import Vina
import pandas as pd

from multiprocessing import Pool




# ---------------------------------------------------------------------------
# Box resolution
# ---------------------------------------------------------------------------

def resolve_box(
    box_config: Path | None,
    box_center: list[float] | None,
    box_size: list[float] | None,
) -> tuple[list[float], list[float]]:
    if box_config is not None:
        if box_center is not None or box_size is not None:
            raise ValueError("Use either --box-config or --box-center/--box-size")
        cfg = json.loads(box_config.read_text(encoding="utf-8"))
        return cfg["center"], cfg["size"]
    if box_center is None or box_size is None:
        raise ValueError("Provide --box-config or both --box-center and --box-size")
    return box_center, box_size


# ---------------------------------------------------------------------------
# Ligand resolution
# ---------------------------------------------------------------------------

def resolve_specs(args: argparse.Namespace) -> list[Path]:
    if args.ligand_pdbqt is not None:
        if args.ligand_pdbqt.suffix.lower() != ".pdbqt":
            raise ValueError("--ligand-pdbqt must be a .pdbqt file")
        return [args.ligand_pdbqt]
    if args.ligand_dir is None:
        raise ValueError("Provide --ligand-pdbqt or --ligand-dir")
    pattern = args.ligand_dir.rglob("*.pdbqt") if not args.no_recursive else args.ligand_dir.glob("*.pdbqt")
    specs = sorted(pattern)
    if not specs:
        raise ValueError(f"No .pdbqt files under {args.ligand_dir}")
    return specs


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------

def dock_one(
    ligand: Path, *,
    output_dir: Path, receptor_pdbqt: Path,
    box_center: list[float], box_size: list[float],
    score_function: str, cpu: int, seed: int, no_refine: bool,
    exhaustiveness: int, n_poses: int, min_rmsd: float, energy_range: float, max_evals: int,
    verbose: bool, skip_existing: bool,
) -> dict[str, object]:
    ligand_id = ligand.stem
    if ligand_id in (".", ".."):
        raise ValueError(f"Invalid ligand filename: {ligand.name}")
    work_dir = output_dir / ligand_id
    work_dir.mkdir(parents=True, exist_ok=True)
    poses_pdbqt = work_dir / f"{ligand_id}_out.pdbqt"
    summary_json = work_dir / "summary.json"

    if skip_existing and summary_json.exists() and poses_pdbqt.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        payload["status"] = "skipped"
        return payload

    if ligand.suffix.lower() != ".pdbqt":
        raise ValueError("run_vina.py only accepts prepared .pdbqt inputs")

    v = Vina(sf_name=score_function, cpu=cpu, seed=seed, no_refine=no_refine, verbosity=1 if verbose else 0)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_file(str(ligand.resolve()))
    v.compute_vina_maps(center=box_center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses, min_rmsd=min_rmsd, max_evals=max_evals)
    v.write_poses(str(poses_pdbqt), n_poses=n_poses, energy_range=energy_range, overwrite=True)
    energies = v.energies(n_poses=n_poses, energy_range=energy_range)

    summary = {
        "ligand_id": ligand_id, "status": "ok",
        "source": str(ligand), "ligand_pdbqt": str(ligand.resolve()),
        "poses_pdbqt": str(poses_pdbqt), "receptor_pdbqt": str(receptor_pdbqt),
        "score_function": score_function,
        "box_center": box_center, "box_size": box_size,
        "best_score": float(energies[0][0]), "pose_count": len(energies),
        "energies": [[float(v) for v in row] for row in energies.tolist()],
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# 内部异常捕获机制
def _worker(payload):
    spec, kw = payload
    try:
        return spec, dock_one(spec, **kw), None
    except Exception as exc:
        return spec, None, repr(exc)



def main(args: argparse.Namespace) -> int:
    if min(args.exhaustiveness, args.n_poses, args.jobs) < 1:
        raise ValueError("--exhaustiveness, --n-poses and --jobs must be >= 1")
    receptor_pdbqt = args.receptor_pdbqt.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    center, size = resolve_box(args.box_config, args.box_center, args.box_size)
    specs = resolve_specs(args)
    jobs = min(args.jobs, len(specs))

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

    def _record(ligand: Path, outcome: dict | None = None, error: Exception | None = None) -> dict[str, int]:
        if outcome is not None:
            results.append(outcome)
        else:
            results.append({"ligand_id": ligand.stem, "status": "failed", "source": str(ligand), "pose_count": "", "error": str(error)})
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
        pbar = tqdm(total=len(specs), desc="Docking", unit="lig") if show_progress else None
        with Pool(processes=jobs, maxtasksperchild=1) as pool:
            for spec, outcome, err in pool.imap_unordered(
                _worker, ((s, task_kw) for s in specs)
            ):
                if err is None:
                    c = _record(spec, outcome=outcome)
                else:
                    c = _record(spec, error=RuntimeError(err))
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(**c)
        if pbar:
            pbar.close()


    results.sort(key=lambda r: str(r["ligand_id"]))
    pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)

    for row in results:
        if row["status"] in ("ok", "skipped"):
            print(f"{row['ligand_id']}\t{row['status']}\t{row.get('best_score', '')}\t{row.get('poses_pdbqt', '')}")
        else:
            print(f"FAIL\t{row['ligand_id']}\t{row.get('error', '')}", file=sys.stderr)

    return 0 if all(r["status"] in ("ok", "skipped") for r in results) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receptor-pdbqt", type=Path, required=True, help="Prepared receptor PDBQT from prepare_receptor.py.")
    p.add_argument("--box-config", type=Path, help="Box JSON from prepare_receptor.py.")
    p.add_argument("--box-center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--box-size", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))

    p.add_argument("--ligand-pdbqt", type=Path, help="Single ligand PDBQT.")
    p.add_argument("--ligand-dir", type=Path, help="Directory of prepared .pdbqt files.")

    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for docking results.")
    p.add_argument("--score-function", choices=("vina", "vinardo"), default="vina")
    p.add_argument("--cpu", type=int, default=8)                                    # cpu数量和exhaustiveness 要相称，有1个cpu就走一次MC计算
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--n-poses", type=int, default=9)
    p.add_argument("--min-rmsd", type=float, default=1.0)
    p.add_argument("--energy-range", type=float, default=3.0)
    p.add_argument("--max-evals", type=int, default=0)
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--jobs", type=int, default=1, help="Parallel docking workers.")
    try:
        raise SystemExit(main(p.parse_args()))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)


# 这里是构建好了一个vina docking
# python -m vina_dock.run_vina --receptor-pdbqt /data3/lpw/dock_workdir/vina/receptor/3A1D_protein.pdbqt --box-config /data3/lpw/dock_workdir/vina/receptor/3A1D_protein.box.json --ligand-dir /data3/lpw/dock_workdir/vina/ligand --output-dir /data3/lpw/dock_workdir/vina/result