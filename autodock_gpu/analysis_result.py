#!/usr/bin/env python3
"""Analyze AutoDock-GPU docking logs (.dlg): energies, poses and RMSD to the GT ligand.

Everything is read from <input-dir>/**/<id>.dlg. Every run of every ligand is
collected (energy + coordinates). With --gt-ligand-dir each pose is compared to
<gt-ligand-dir>/<id>/<id>_ligand.sdf (heavy atoms, symmetry-corrected, no alignment).

Outputs:
    runs.csv        one row per ligand per run: rank, run, cluster, energy, rmsd
    summary.csv     one row per ligand: best energy, RMSD of the top-1 pose and of the closest run
    poses/<id>.sdf  (--export-poses) every pose of the ligand in rank order (top-1 first), heavy atoms only

rank is the position of a run in the DLG's RMSD TABLE: clusters one after another,
runs inside a cluster by energy. Rank 1 is the lowest-energy run overall; later
ranks are not a strict global energy order.

Example:
    python -m autodock_gpu.analysis_result \\
        --input-dir /path/to/dock_result \\
        --gt-ligand-dir /path/to/bonded \\
        --output-dir /path/to/analysis --export-poses
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from tqdm import tqdm

from utils.file_utils import load_mol


INPUT_PREFIX = "INPUT-LIGAND-PDBQT: "       # 输入配体的原文，只出现一次
DOCKED_PREFIX = "DOCKED: "                  # 每次 run 的结果，每个 MODEL 一段


def parse_dlg(dlg_path):
    """Read one DLG. Returns (runs, smiles, index_map).

    runs:      one dict per run {run, energy, coords, rank, cluster}, sorted by rank (runs[0] is the top-1 pose).
               coords holds the xyz of every ATOM line of that run, in file order.
    smiles:    the ligand as Meeko wrote it into the PDBQT.
    index_map: {SMILES atom index: position in coords}, both 0-based.
    """
    runs = []
    current = None              # 正在读的那个 MODEL
    smiles = None
    idx_numbers = []
    ranking = {}                # run -> (rank, cluster)

    for line in Path(dlg_path).read_text(errors="replace").splitlines():
        if line.startswith(INPUT_PREFIX):
            body = line[len(INPUT_PREFIX):]
            if body.startswith("REMARK SMILES IDX"):                # REMARK SMILES IDX 1 1 2 2 16 8 ...
                idx_numbers.extend(int(x) for x in body.split()[3:])
            elif body.startswith("REMARK SMILES"):                  # REMARK SMILES CC(=O)N1CCN...
                smiles = body.split()[2]

        elif line.startswith(DOCKED_PREFIX):
            body = line[len(DOCKED_PREFIX):]
            if body.startswith("MODEL"):
                current = {"coords": []}
            elif body.startswith("USER    Run ="):                  # USER    Run = 6
                current["run"] = int(body.split("=")[1])
            elif body.startswith("USER    Estimated Free Energy of Binding"):   # ... =  -6.43 kcal/mol  [...]
                current["energy"] = float(body.split("=")[1].split()[0])
            elif body.startswith(("ATOM", "HETATM")):
                # 坐标是 PDB 固定列宽，必须按列切片；用 split 会在 "55.069-152.593" 这种粘连处出错
                current["coords"].append((float(body[30:38]), float(body[38:46]), float(body[46:54])))
            elif body.startswith("ENDMDL"):
                runs.append(current)
                current = None

        elif line.rstrip().endswith("RANKING"):
            # RMSD TABLE 的一行：Rank  SubRank  Run  Energy  ClusterRMSD  RefRMSD  RANKING
            # 一个聚类接一个聚类地列出，聚类内按能量从低到高；第一行是全局能量最低的 run
            fields = line.split()
            ranking[int(fields[2])] = (len(ranking) + 1, int(fields[0]))

    if not runs:
        raise ValueError("no DOCKED models in DLG")
    if smiles is None:
        raise ValueError("no REMARK SMILES in DLG (ligand PDBQT not written by Meeko?)")
    if set(ranking) != {r["run"] for r in runs}:
        raise ValueError("RMSD TABLE does not list every run")

    for r in runs:
        r["rank"], r["cluster"] = ranking[r["run"]]
    runs.sort(key=lambda r: r["rank"])

    # IDX 里的数字两两一组：(SMILES 原子序号, PDBQT 原子序号)，都从 1 开始
    index_map = {idx_numbers[k] - 1: idx_numbers[k + 1] - 1 for k in range(0, len(idx_numbers), 2)}
    return runs, smiles, index_map


def build_pose_mol(smiles, index_map, runs):
    """Heavy-atom RDKit mol of the ligand with one conformer per run, in the order of `runs`."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot read the SMILES in the DLG: {smiles}")
    if set(index_map) != set(range(mol.GetNumAtoms())):
        raise ValueError("REMARK SMILES IDX does not cover every heavy atom")
    for r in runs:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for smiles_idx, pdbqt_idx in index_map.items():
            conf.SetAtomPosition(smiles_idx, r["coords"][pdbqt_idx])
        mol.AddConformer(conf, assignId=True)
    return mol


def pose_rmsds(pose_mol, gt_mol):
    """Symmetry-corrected heavy-atom RMSD of every pose to the GT ligand, without alignment."""
    gt = Chem.RemoveHs(gt_mol)
    return [rdMolAlign.CalcRMS(pose_mol, gt, prbId=conf.GetId()) for conf in pose_mol.GetConformers()]


def write_poses_sdf(mol, ligand_id, runs, output_path):
    """Write every pose to one SDF, in the order of `runs`."""
    with Chem.SDWriter(str(output_path)) as w:
        for conf, r in zip(mol.GetConformers(), runs):
            mol.SetProp("_Name", f"{ligand_id}_run{r['run']}")
            mol.SetProp("run", str(r["run"]))
            mol.SetProp("energy", f"{r['energy']:.2f}")
            if "rmsd" in r:
                mol.SetProp("rmsd", f"{r['rmsd']:.4f}")
            w.write(mol, confId=conf.GetId())


def analyze_dlg(dlg_path, gt_ligand_dir=None, pose_dir=None):
    """Analyze one ligand. Returns one row (dict) per run, in rank order (top-1 first)."""
    dlg_path = Path(dlg_path)
    ligand_id = dlg_path.stem
    runs, smiles, index_map = parse_dlg(dlg_path)

    if gt_ligand_dir is not None or pose_dir is not None:
        mol = build_pose_mol(smiles, index_map, runs)
        if gt_ligand_dir is not None:
            gt_mol = load_mol(str(Path(gt_ligand_dir) / ligand_id / f"{ligand_id}_ligand.sdf"))
            for r, rmsd in zip(runs, pose_rmsds(mol, gt_mol)):
                r["rmsd"] = rmsd
        if pose_dir is not None:
            write_poses_sdf(mol, ligand_id, runs, Path(pose_dir) / f"{ligand_id}.sdf")

    return [{"ligand_id": ligand_id, "rank": r["rank"], "run": r["run"], "cluster": r["cluster"],
             "energy": r["energy"], "rmsd": r.get("rmsd")} for r in runs]


def summarize(runs_df):
    """One row per ligand: the top-1 pose (rank 1) and, if RMSD is available, the run closest to the GT ligand."""
    top1 = runs_df[runs_df["rank"] == 1].set_index("ligand_id")
    summary = pd.DataFrame({
        "num_runs": runs_df.groupby("ligand_id").size(),
        "best_energy": top1["energy"],
        "best_run": top1["run"],
    })
    if "rmsd" in runs_df:
        closest = runs_df.loc[runs_df.groupby("ligand_id")["rmsd"].idxmin()].set_index("ligand_id")
        summary["rmsd_top1"] = top1["rmsd"]
        summary["rmsd_min"] = closest["rmsd"]
        summary["rmsd_min_run"] = closest["run"]
    return summary.reset_index()


def print_rmsd_summary(summary, threshold):
    print(f"\n=== RMSD to GT ligand ({len(summary)} ligands, heavy atoms, no alignment) ===")
    for label, column in (("top-1 pose", "rmsd_top1"), ("closest run", "rmsd_min")):
        values = summary[column]
        hits = int((values < threshold).sum())
        print(f"{label:<12} median {values.median():6.2f} A   "
              f"< {threshold:g} A: {hits}/{len(values)} ({100 * hits / len(values):.1f}%)")


def main(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = output_dir / "poses" if args.export_poses else None
    if pose_dir is not None:
        pose_dir.mkdir(exist_ok=True)

    dlg_files = sorted(Path(args.input_dir).rglob("*.dlg"))
    if not dlg_files:
        raise FileNotFoundError(f"No .dlg files in {args.input_dir}")

    rows = []
    failures = []
    for dlg in tqdm(dlg_files, desc="Analyzing DLG"):
        try:
            rows.extend(analyze_dlg(dlg, args.gt_ligand_dir, pose_dir))
        except Exception as exc:
            failures.append((dlg.stem, str(exc)))

    for ligand_id, msg in failures:
        print(f"FAIL\t{ligand_id}\t{msg}", file=sys.stderr)
    print(f"\nDone. OK: {len(dlg_files) - len(failures)}, Failed: {len(failures)}", file=sys.stderr)
    if not rows:
        return 1

    runs_df = pd.DataFrame(rows).dropna(axis=1, how="all")      # 没给 GT 时去掉空的 rmsd 列
    summary = summarize(runs_df)
    runs_df.to_csv(output_dir / "runs.csv", index=False, float_format="%.4f")
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.4f")
    print(f"Wrote {output_dir / 'runs.csv'}", file=sys.stderr)
    print(f"Wrote {output_dir / 'summary.csv'}", file=sys.stderr)
    if pose_dir is not None:
        print(f"Wrote poses to {pose_dir}", file=sys.stderr)

    if "rmsd_top1" in summary:
        print_rmsd_summary(summary, args.rmsd_threshold)
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",      type=Path, required=True, help="Directory of AutoDock-GPU .dlg files (searched recursively).")
    parser.add_argument("--output-dir",     type=Path, required=True, help="Output directory for CSVs and poses.")
    parser.add_argument("--gt-ligand-dir",  type=Path, default=None, help="GT ligands as <id>/<id>_ligand.sdf; enables RMSD analysis.")
    parser.add_argument("--export-poses",   action="store_true", help="Write every docked pose to <output-dir>/poses/<id>.sdf.")
    parser.add_argument("--rmsd-threshold", type=float, default=2.0, help="Success cutoff (A) for the RMSD summary.")
    raise SystemExit(main(parser.parse_args()))


# 调试
# python -m autodock_gpu.analysis_result --input-dir /data3/lpw/dock_workdir/autodock_gpu/cov200_test/dock_result --gt-ligand-dir /data3/lpw/data/covdocker/data/processed/bonded --output-dir /data3/lpw/dock_workdir/autodock_gpu/cov200_test/analysis --export-poses
