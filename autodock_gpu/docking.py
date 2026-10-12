#!/usr/bin/env python3
"""Run AutoDock-GPU docking in single or batch mode.

Modes:
  Single ligand:  --ligand <pdbqt> --fld <fld>
  Batch ligands:  --ligand-dir <dir> --fld <fld>
  Single DPF:     --dpf <dpf>            (CPU-pipeline compatibility)
  Batch DPFs:     --dpf-dir <dir>        (CPU-pipeline compatibility)

Note on --num_workers: autodock_gpu uses one GPU device. Setting workers > 1
only helps if you have multiple GPUs (each worker pinned to a different
--device) or genuinely free VRAM headroom; otherwise leave at 1.
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from utils.file_utils import load_idx_list
from .config import (ADGPU_BINARY, CONDA_PREFIX, DEFAULT_DEVICE,
                     DEFAULT_NRUN, DEFAULT_NEV, DEFAULT_HEURISTICS)


SUMMARY_KEYWORDS = ("Estimated Free Energy of Binding", "USER    Run =", "USER    Cluster Rank =")


def make_env(conda_prefix):
    env = dict(os.environ)
    lib = str(Path(conda_prefix).resolve() / "lib")
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib}:{ld}" if ld else lib
    return env


def run_subprocess(cmd, cwd, conda_prefix):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), env=make_env(conda_prefix))
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "AutoDock-GPU failed")








def run_autodock_gpu(ligand_pdbqt, fld_path, output_dir, *,
                     binary=ADGPU_BINARY, conda_prefix=CONDA_PREFIX,
                     device=DEFAULT_DEVICE, nrun=DEFAULT_NRUN,
                     nev=DEFAULT_NEV, heuristics=DEFAULT_HEURISTICS):
    """Run autodock_gpu on one ligand against one .fld grid. Returns dlg path."""
    ligand_pdbqt = Path(ligand_pdbqt)
    fld_path = Path(fld_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = ligand_pdbqt.stem
    cmd = [
        str(binary),
        "--lfile", str(ligand_pdbqt),
        "--ffile", str(fld_path),
        "--resnam", str(output_dir / name),
        "--devnum", str(device),
        "--nrun", str(nrun),
        "--nev", str(nev),
        "--heuristics", str(heuristics),
        "--xmloutput", "1",
        "--dlgoutput", "1",
    ]
    run_subprocess(cmd, output_dir, conda_prefix)
    return str(output_dir / f"{name}.dlg")

# dpf是autodock预处理的输入文件
def run_autodock_gpu_dpf(dpf_path, output_dir, *,
                         binary=ADGPU_BINARY, conda_prefix=CONDA_PREFIX,
                         device=DEFAULT_DEVICE):
    """Run autodock_gpu using DPF import (CPU-pipeline compatibility)."""
    dpf_path = Path(dpf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = dpf_path.stem
    cmd = [
        str(binary),
        "--import_dpf", str(dpf_path),
        "--resnam", str(output_dir / name),
        "--devnum", str(device),
        "--xmloutput", "1",
        "--dlgoutput", "1",
    ]
    run_subprocess(cmd, output_dir, conda_prefix)
    return str(output_dir / f"{name}.dlg")


def parse_dlg_summary(dlg_path):
    dlg_path = Path(dlg_path)
    if not dlg_path.exists():
        return []
    return [l.strip() for l in dlg_path.read_text(errors="replace").splitlines()
            if any(k in l for k in SUMMARY_KEYWORDS)]






def collect_paired_jobs(ligand_dir, grid_dir, data_list, output_dir, skip_existing):
    jobs, missing = [], []
    for pdb_id in load_idx_list(data_list):
        lig = Path(ligand_dir) / f"{pdb_id}.pdbqt"
        name = f"{pdb_id}_protein_pocket10"
        fld = Path(grid_dir) / name / f"{name}.maps.fld"
        out = Path(output_dir) / pdb_id
        if not (lig.exists() and fld.exists()):
            missing.append(pdb_id)
            continue
        if skip_existing and (out / f"{pdb_id}.dlg").exists():
            continue
        jobs.append((lig, fld, out))
    return jobs, missing




def collect_ligand_jobs(ligand_dir, fld_path, output_dir, skip_existing, pattern="*.pdbqt"):
    ligand_dir = Path(ligand_dir)
    output_dir = Path(output_dir)
    fld_path = Path(fld_path)
    ligands = sorted(ligand_dir.rglob(pattern))
    if not ligands:
        raise FileNotFoundError(f"No {pattern} in {ligand_dir}")

    jobs = []
    for lig in ligands:
        dlg = output_dir / f"{lig.stem}.dlg"
        if skip_existing and dlg.exists():
            continue
        jobs.append((lig, fld_path, output_dir))
    return jobs


def collect_dpf_jobs(dpf_dir, output_dir, skip_existing):
    dpf_dir = Path(dpf_dir)
    output_dir = Path(output_dir)
    dpfs = sorted(dpf_dir.rglob("*.dpf"))
    if not dpfs:
        raise FileNotFoundError(f"No .dpf in {dpf_dir}")
    jobs = []
    for dpf in dpfs:
        dlg = output_dir / f"{dpf.stem}.dlg"
        if skip_existing and dlg.exists():
            continue
        jobs.append((dpf, output_dir))
    return jobs


def run_batch_ligands(jobs, num_workers, *, binary, conda_prefix,
                      device, nrun, nev, heuristics):
    if not jobs:
        return {}, []

    success = {}
    failures = []
    if num_workers == 1:
        for ligand_pdbqt, fld_path, out_dir in tqdm(jobs, desc="Running AutoDock-GPU"):
            try:
                dlg = run_autodock_gpu(ligand_pdbqt, fld_path, out_dir,
                                       binary=binary, conda_prefix=conda_prefix,
                                       device=device, nrun=nrun, nev=nev, heuristics=heuristics)
                success[ligand_pdbqt.stem] = dlg
            except Exception as exc:
                failures.append((ligand_pdbqt.stem, str(exc)))
    else:
        with tqdm_joblib(total=len(jobs), desc="Running AutoDock-GPU"):
            results = Parallel(n_jobs=num_workers, return_exceptions=True)(
                delayed(run_autodock_gpu)(ligand_pdbqt, fld_path, out_dir,
                                          binary=binary, conda_prefix=conda_prefix,
                                          device=device, nrun=nrun, nev=nev, heuristics=heuristics)
                for ligand_pdbqt, fld_path, out_dir in jobs
            )
        for (ligand_pdbqt, _, _), result in zip(jobs, results):
            if isinstance(result, Exception):
                failures.append((ligand_pdbqt.stem, str(result)))
            else:
                success[ligand_pdbqt.stem] = result
    return success, failures


def run_batch_dpfs(jobs, num_workers, *, binary, conda_prefix, device):
    if not jobs:
        return {}, []

    success = {}
    failures = []
    if num_workers == 1:
        for dpf_path, out_dir in tqdm(jobs, desc="Running AutoDock-GPU (DPF)"):
            try:
                dlg = run_autodock_gpu_dpf(dpf_path, out_dir,
                                           binary=binary, conda_prefix=conda_prefix, device=device)
                success[dpf_path.stem] = dlg
            except Exception as exc:
                failures.append((dpf_path.stem, str(exc)))
    else:
        with tqdm_joblib(total=len(jobs), desc="Running AutoDock-GPU (DPF)"):
            results = Parallel(n_jobs=num_workers, return_exceptions=True)(
                delayed(run_autodock_gpu_dpf)(dpf_path, out_dir,
                                              binary=binary, conda_prefix=conda_prefix, device=device)
                for dpf_path, out_dir in jobs
            )
        for (dpf_path, _), result in zip(jobs, results):
            if isinstance(result, Exception):
                failures.append((dpf_path.stem, str(result)))
            else:
                success[dpf_path.stem] = result
    return success, failures


def main(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = Path(args.binary).resolve()
    conda_prefix = Path(args.conda_prefix).resolve()

    # Single ligand
    if args.ligand is not None:
        if args.fld is None:
            raise ValueError("--fld is required with --ligand")
        ligand = Path(args.ligand).resolve()
        fld = Path(args.fld).resolve()
        dlg = run_autodock_gpu(ligand, fld, output_dir,
                               binary=binary, conda_prefix=conda_prefix,
                               device=args.device, nrun=args.nrun,
                               nev=args.nev, heuristics=args.heuristics)
        for line in parse_dlg_summary(dlg)[:12]:
            print(line)
        return 0

    # Single DPF
    if args.dpf is not None:
        dpf = Path(args.dpf).resolve()
        dlg = run_autodock_gpu_dpf(dpf, output_dir,
                                   binary=binary, conda_prefix=conda_prefix, device=args.device)
        for line in parse_dlg_summary(dlg)[:12]:
            print(line)
        return 0

    # Batch ligand-dir
    elif args.ligand_dir is not None and args.grid_dir is None:
        if args.fld is None:
            raise ValueError("--fld is required with --ligand-dir")
        jobs = collect_ligand_jobs(args.ligand_dir, args.fld, output_dir,
                                    args.skip_existing, args.ligand_pattern)
        success, failures = run_batch_ligands(jobs, args.num_workers,
                                              binary=binary, conda_prefix=conda_prefix,
                                              device=args.device, nrun=args.nrun,
                                              nev=args.nev, heuristics=args.heuristics)
    elif args.dpf_dir is not None:
        jobs = collect_dpf_jobs(args.dpf_dir, output_dir, args.skip_existing)
        success, failures = run_batch_dpfs(jobs, args.num_workers,
                                            binary=binary, conda_prefix=conda_prefix,
                                            device=args.device)

    # ligand-pocket pair'
    elif args.ligand_dir and args.grid_dir and args.data_list:
        jobs, missing = collect_paired_jobs(args.ligand_dir, args.grid_dir, args.data_list, output_dir, args.skip_existing)
        success, failures = run_batch_ligands(jobs, args.num_workers,
                                              binary=binary, conda_prefix=conda_prefix,
                                              device=args.device, nrun=args.nrun,
                                              nev=args.nev, heuristics=args.heuristics)
        
        if missing:
            print(f"Missing ligand/grid pairs for {len(missing)} entries in {args.data_list}: {', '.join(missing)}", file=sys.stderr)
        if not jobs:
            print("No jobs to run after filtering for missing pairs and existing outputs.", file=sys.stderr)


        for name, path in success.items():
            print(f"OK\t{name}\t{path}")
        for name, msg in failures:
            print(f"FAIL\t{name}\t{msg}", file=sys.stderr)
        print(f"\nDone. OK: {len(success)}, Failed: {len(failures)}", file=sys.stderr)
        return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ligand",         type=Path, default=None, help="Single ligand PDBQT.")

    parser.add_argument("--ligand-pattern", default="*.pdbqt", help="Glob pattern under --ligand-dir.")
    parser.add_argument("--fld",            type=Path, default=None, help="Grid .maps.fld file.")
    parser.add_argument("--dpf",            type=Path, default=None, help="Single DPF (CPU-pipeline compat).")
    parser.add_argument("--dpf-dir",        type=Path, default=None, help="Directory of DPFs (CPU-pipeline compat).")
    parser.add_argument("--output-dir",     type=Path, default=None, help="Output directory.")
    parser.add_argument("--ligand-dir",     type=Path, default=None, help="Directory of ligand PDBQTs.")
    parser.add_argument("--grid-dir",       type=Path, default=None, help="Directory of grid .maps.fld files.")
    parser.add_argument("--data-list",      type=Path, default=None, help="List of ligand/grid pairs (one per line).")
    
    parser.add_argument("--binary",         type=Path, default=ADGPU_BINARY)
    parser.add_argument("--conda-prefix",   type=Path, default=CONDA_PREFIX)
    parser.add_argument("--device",         type=int, default=DEFAULT_DEVICE)
    parser.add_argument("--nrun",           type=int, default=DEFAULT_NRUN)
    parser.add_argument("--nev",            type=int, default=DEFAULT_NEV)
    parser.add_argument("--heuristics",     type=int, choices=(0, 1), default=DEFAULT_HEURISTICS)
    parser.add_argument("--skip-existing",  action="store_true", help="Skip jobs whose .dlg already exists.")
    parser.add_argument("--num_workers",    type=int, default=2, help="Parallel workers (batch mode). See module docstring.")
    raise SystemExit(main(parser.parse_args()))



# output_dir  /data3/lpw/dock_workdir/autodock_gpu/cov200_test/dock_result


# 调试
# python -m autodock_gpu.docking --ligand /data3/lpw/dock_workdir/autodock_gpu/cov200_test/lig_pdbqt/5REJ.pdbqt --fld  /data3/lpw/dock_workdir/autodock_gpu/cov200_test/grid/5REJ_protein_pocket10/5REJ_protein_pocket10.maps.fld --output-dir  /data3/lpw/dock_workdir/autodock_gpu/cov200_test/dock_result

# pair docking
# python -m autodock_gpu.docking --ligand-dir /data3/lpw/dock_workdir/autodock_gpu/cov200_test/lig_pdbqt --grid-dir  /data3/lpw/dock_workdir/autodock_gpu/cov200_test/grid --output-dir  /data3/lpw/dock_workdir/autodock_gpu/cov200_test/dock_result --data-list /home/lpw/dock_scripts/autodock/data/cov200_list.txt

