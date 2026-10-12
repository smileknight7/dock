#!/usr/bin/env python3
"""Generate GPF/DPF files and run AutoGrid4 for AutoDock4.

Receptor input may be:
    --receptor-pdb    PDB file; this script converts it to PDBQT via MGLTools.
    --receptor-pdbqt  Already-prepared PDBQT (skip MGLTools step).

Box info must be provided via:
    --reference-ligand        compute box center and size from ligand bounding box
    --box-center + --box-size explicit box (Angstrom)
"""

from __future__ import annotations
import argparse
import math
import shutil
import sys
from pathlib import Path
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import pandas as pd

from .config import (GRID_SPACING, PREPARE_DPF4, PREPARE_GPF4, PREPARE_RECEPTOR4,
                     run_autogrid4, run_mgltools_script)
from utils.pocket_utils import load_ligand_coords, compute_box_bbox


def parse_triplet(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"Expected three values, got: {value}")
        return tuple(float(x) for x in value)
    parts = [float(x.strip()) for x in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values, got: {value}")
    return tuple(parts)


def box_size_to_npts(box_size):
    npts = []
    for size in box_size:
        count = int(math.ceil(size / GRID_SPACING))
        if count % 2 != 0:
            count += 1
        npts.append(count)
    return tuple(npts)


def copy_if_needed(src, dst):
    if Path(src).resolve() != Path(dst).resolve():
        shutil.copy2(src, dst)


def prepare_receptor_pdbqt(receptor_pdb, output_pdbqt):
    """Convert receptor PDB -> PDBQT via MGLTools prepare_receptor4.py."""
    output_pdbqt = Path(output_pdbqt)
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    run_mgltools_script(
        PREPARE_RECEPTOR4,
        ["-r", str(receptor_pdb), "-o", str(output_pdbqt)],
        Path(receptor_pdb).parent,
    )


def generate_gpf(ligand_pdbqt, receptor_pdbqt, center, npts, output_gpf):
    args = [
        "-l", str(ligand_pdbqt),
        "-r", str(receptor_pdbqt),
        "-o", str(output_gpf),
        "-p", f"gridcenter={center[0]:.3f},{center[1]:.3f},{center[2]:.3f}",
    ]
    if npts is not None:
        args.extend(["-p", f"npts={npts[0]},{npts[1]},{npts[2]}"])
    run_mgltools_script(PREPARE_GPF4, args, Path(receptor_pdbqt).parent)


def generate_dpf(ligand_pdbqt, receptor_pdbqt, output_dpf):
    run_mgltools_script(
        PREPARE_DPF4,
        ["-l", str(ligand_pdbqt), "-r", str(receptor_pdbqt), "-o", str(output_dpf)],
        Path(receptor_pdbqt).parent,
    )


def run_grid(ligand_pdbqt, receptor_pdbqt, output_dir, box_center, box_size):                   # 新生成的文件
    """Generate GPF + DPF for one ligand and run autogrid4."""
    npts = box_size_to_npts(box_size)

    local_ligand = output_dir / ligand_pdbqt.name
    local_receptor = output_dir / receptor_pdbqt.name
    copy_if_needed(ligand_pdbqt, local_ligand)
    copy_if_needed(receptor_pdbqt, local_receptor)

    gpf_path = output_dir / f"{local_ligand.stem}_{local_receptor.stem}.gpf"
    dpf_path = output_dir / f"{local_ligand.stem}_{local_receptor.stem}.dpf"
    glg_path = gpf_path.with_suffix(".glg")

    generate_gpf(local_ligand, local_receptor, box_center, npts, gpf_path)
    generate_dpf(local_ligand, local_receptor, dpf_path)
    run_autogrid4(gpf_path, glg_path)


def collect_jobs(ligand_csv, id_column, receptor_pdbqt, ligand_dir, out_dir, skip_existing):
    df = pd.read_csv(ligand_csv)
    ligand_ids = [str(row[id_column]) if id_column else f'ligand_{i:04d}' for i, row in df.iterrows()]

    ligand_dir = Path(ligand_dir)
    out_dir = Path(out_dir)

    jobs = []
    for ligand in ligand_ids:
        ligand_file = ligand_dir / f"{ligand}.pdbqt"
        glg_file = out_dir / f"{ligand_file.stem}_{receptor_pdbqt.stem}.glg"
        if skip_existing and glg_file.exists():
            continue
        jobs.append((ligand_file, receptor_pdbqt, out_dir))
    return jobs


def run_batch(ligand_csv, id_column, ligand_dir, receptor_pdbqt, output_dir,
              box_center, box_size, num_workers):
    jobs = collect_jobs(ligand_csv, id_column, receptor_pdbqt, ligand_dir, output_dir, skip_existing=True)
    if not jobs:
        return {}, []

    success = {}
    failures = []
    if num_workers == 1:
        for ligand_file, receptor_pdbqt, out_dir in tqdm(jobs, desc="Processing ligands"):
            try:
                run_grid(ligand_file, receptor_pdbqt, out_dir, box_center, box_size)
                success[ligand_file.stem] = str(out_dir)
            except Exception as exc:
                failures.append((ligand_file.stem, str(exc)))
    else:
        with tqdm_joblib(total=len(jobs), desc="Processing ligands"):
            results = Parallel(n_jobs=num_workers, return_exceptions=True)(
                delayed(run_grid)(ligand_file, receptor_pdbqt, out_dir, box_center, box_size)
                for ligand_file, receptor_pdbqt, out_dir in jobs
            )
        for (ligand_file, _, out_dir), result in zip(jobs, results):
            if isinstance(result, Exception):
                failures.append((ligand_file.stem, str(result)))
            else:
                success[ligand_file.stem] = str(out_dir)

    return success, failures


def resolve_box(args):
    """Compute (box_center, box_size) from CLI args. Exactly one source must be given."""
    has_ref = args.reference_ligand is not None
    has_box = args.box_center is not None or args.box_size is not None
    if has_ref == has_box:
        raise ValueError("Use either --reference-ligand or --box-center+--box-size (not both, not neither)")
    if (args.box_center is None) != (args.box_size is None):
        raise ValueError("--box-center and --box-size must be used together")

    if has_ref:
        coords = load_ligand_coords(Path(args.reference_ligand).resolve())
        return compute_box_bbox(coords, args.padding)
    return parse_triplet(args.box_center), parse_triplet(args.box_size)


def resolve_receptor_pdbqt(args, output_dir):
    """Return a Path to the receptor PDBQT, converting from PDB if needed."""
    if args.receptor_pdbqt is not None:
        return Path(args.receptor_pdbqt).resolve()
    receptor_pdb = Path(args.receptor_pdb).resolve()
    receptor_pdbqt = output_dir / f"{receptor_pdb.stem}.pdbqt"
    print(f"Preparing receptor: {receptor_pdb.name} -> {receptor_pdbqt.name}", file=sys.stderr)
    prepare_receptor_pdbqt(receptor_pdb, receptor_pdbqt)
    return receptor_pdbqt


def main(args):
    if (args.receptor_pdb is None) == (args.receptor_pdbqt is None):
        raise ValueError("Provide exactly one of --receptor-pdb or --receptor-pdbqt")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    box_center, box_size = resolve_box(args)
    print(f"Box center: ({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f})", file=sys.stderr)
    print(f"Box size:   ({box_size[0]:.3f}, {box_size[1]:.3f}, {box_size[2]:.3f})", file=sys.stderr)

    receptor_pdbqt = resolve_receptor_pdbqt(args, output_dir)

    if args.ligand_pdbqt is not None:
        ligand_pdbqt = Path(args.ligand_pdbqt).resolve()
        run_grid(ligand_pdbqt, receptor_pdbqt, output_dir, box_center, box_size)
        return 0

    if args.ligand_csv is not None:
        ligand_csv = Path(args.ligand_csv).resolve()
        ligand_dir = Path(args.ligand_dir).resolve()
        success, failures = run_batch(
            ligand_csv, args.id_column, ligand_dir, receptor_pdbqt,
            output_dir, box_center, box_size, num_workers=args.num_workers,
        )
        for name, path in success.items():
            print(f"OK\t{name}\t{path}")
        for name, msg in failures:
            print(f"FAIL\t{name}\t{msg}", file=sys.stderr)
        print(f"\nDone. OK: {len(success)}, Failed: {len(failures)}", file=sys.stderr)
        return 1 if failures else 0

    print("Either --ligand-pdbqt or --ligand-csv must be provided.")
    return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ligand-pdbqt",     type=Path, help="Single ligand PDBQT (single-ligand mode).")
    parser.add_argument("--id-column",        type=str, help="Column in ligand CSV used as ligand ID.")
    parser.add_argument("--ligand-csv",       type=Path, default='/data3/lpw/data/dock_data/zinc_50.csv', help="CSV file containing a SMILES column.")
    parser.add_argument("--ligand-dir",       type=Path, default='/data3/lpw/dock_workdir/autodock/test/ligands', help="Directory containing ligand PDBQT files.")
    parser.add_argument("--receptor-pdb",     type=Path, default='/data3/lpw/dock_workdir/3I03/3I03_protein.pdb', help="Receptor PDB; auto-converted to PDBQT via MGLTools.")
    parser.add_argument("--receptor-pdbqt",   type=Path, default=None, help="Already-prepared receptor PDBQT.")
    parser.add_argument("--reference-ligand", type=Path, default='/data3/lpw/dock_workdir/3I03/3I03_ligand.sdf', help="Reference ligand for box envelope.")
    parser.add_argument("--box-center",       type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--box-size",         type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--padding",          type=float, default=4.0, help="Padding (A) for --reference-ligand.")
    parser.add_argument("--output-dir",       type=Path, default='/data3/lpw/dock_workdir/autodock/test', help="Output directory.")
    parser.add_argument("--num_workers",      type=int, default=1, help="Parallel workers for batch mode.")
    raise SystemExit(main(parser.parse_args()))
