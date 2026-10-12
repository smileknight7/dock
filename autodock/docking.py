#!/usr/bin/env python3
"""Run AutoDock4 docking in single or batch mode.
修改内容：
1.这里要修改的内容主要是，docking结束之后要保存结果到csv文件夹中，还有添加进度条的功能
2.还有个优化的空间，就是将来joblib部分的func抽提出来，构造一个包装函数

"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from .config import run_autodock4


FINAL_SUMMARY_PREFIXES = (
    "USER    Run =",
    "USER    Cluster Rank =",
    "USER    Number of conformations in this cluster =",
    "USER    (1) Final Intermolecular Energy",
    "USER        vdW + Hbond + desolv Energy",
    "USER        Electrostatic Energy",
    "USER    (2) Final Total Internal Energy",
    "USER    (3) Torsional Free Energy",
    "USER    (4) Unbound System's Energy",
)


def run_autodock(dpf_path, dlg_path):
    run_autodock4(dpf_path, dlg_path)
    return str(dlg_path)


def parse_dlg_results(dlg_path):
    lines = Path(dlg_path).read_text(encoding="utf-8").splitlines()
    current_run = None
    run_energies = []
    final_summary = []
    in_final_model = False

    for line in lines:
        stripped = line.strip()
        if line.startswith("DOCKED: USER    Run ="):
            current_run = int(line.split("=", 1)[1].strip())
        elif current_run is not None and "Estimated Free Energy of Binding" in line and line.startswith("DOCKED: USER"):
            run_energies.append(f"Run {current_run}: {stripped}")
        elif line.startswith("MODEL"):
            in_final_model = True
        elif in_final_model and (stripped.startswith(FINAL_SUMMARY_PREFIXES) or
                                  ("Estimated Free Energy of Binding" in line and line.startswith("USER"))):
            final_summary.append(stripped)
            if stripped.startswith("USER    (4) Unbound System's Energy"):
                break

    summary = list(run_energies)
    if final_summary:
        summary.append("Best cluster summary:")
        summary.extend(final_summary)
    return summary


def collect_jobs(dpf_dir, output_dir, skip_existing):
    dpf_files = sorted(Path(dpf_dir).rglob("*.dpf"))
    if not dpf_files:
        raise FileNotFoundError(f"No .dpf files in {dpf_dir}")

    jobs = []
    for dpf_path in dpf_files:
        out = (output_dir / dpf_path.parent.relative_to(dpf_dir)) if output_dir else dpf_path.parent
        out.mkdir(parents=True, exist_ok=True)
        dlg_path = out / dpf_path.with_suffix(".dlg").name
        if skip_existing and dlg_path.exists():
            print(f"SKIP: {dpf_path.stem}")
            continue
        jobs.append((dpf_path, dlg_path))
    return jobs


def run_batch(dpf_dir, output_dir, num_workers, skip_existing):
    jobs = collect_jobs(dpf_dir, output_dir, skip_existing)
    if not jobs:
        return {}, []

    success = {}
    failures = []
    if num_workers == 1:
        for dpf_path, dlg_path in tqdm(jobs, desc="Running AutoDock4"):
            try:
                run_autodock(dpf_path, dlg_path)
                success[dpf_path.stem] = str(dlg_path)
            except Exception as exc:
                failures.append((dpf_path.stem, str(exc)))
    else:
        with tqdm_joblib(total=len(jobs), desc="Running AutoDock4"):
            results = Parallel(n_jobs=num_workers, return_exceptions=True)(
                delayed(run_autodock)(dpf_path, dlg_path) for dpf_path, dlg_path in jobs
            )
            
        for (dpf_path, dlg_path), result in zip(jobs, results):
            if isinstance(result, Exception):
                failures.append((dpf_path.stem, str(result)))
            else:
                success[dpf_path.stem] = str(dlg_path)

    return success, failures


def main(args):
    if args.dpf is not None:
        dpf_path = Path(args.dpf).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else dpf_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        dlg_path = output_dir / dpf_path.with_suffix(".dlg").name
        run_autodock(dpf_path, dlg_path)
        for line in parse_dlg_results(dlg_path):
            print(line)
        return 0



    dpf_dir = Path(args.dpf_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    success, failures = run_batch(dpf_dir, output_dir, args.num_workers, args.skip_existing)
    for name, path in success.items():
        print(f"OK\t{name}\t{path}")
    for name, msg in failures:
        print(f"FAIL\t{name}\t{msg}", file=sys.stderr)
    print(f"\nDone. OK: {len(success)}, Failed: {len(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # source_group = parser.add_mutually_exclusive_group(required=True)       /data3/lpw/dock_workdir/autodock/test/ligand_0000_3I03_protein_pocket.dpf   
    parser.add_argument("--dpf",     type=Path,  help="Single .dpf file (single mode).")
    parser.add_argument("--dpf-dir", type=Path,default='/data3/lpw/dock_workdir/autodock/test',  help="Directory of .dpf files (batch mode).")
    parser.add_argument("--output-dir",    type=Path, default='/data3/lpw/dock_workdir/autodock/test', help="Output directory.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip jobs whose .dlg already exists.")
    parser.add_argument("--num_workers",   type=int, default=20, help="Parallel workers (batch mode).")
    raise SystemExit(main(parser.parse_args()))

