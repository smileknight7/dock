#!/usr/bin/env python3
"""
AutoDock 批量对接结果分析。

- 解析每个 dlg 文件，汇总成一张 DataFrame
- 输出 summary CSV（所有 ligand）+ top-N TXT（分数最低的 N 个）
- 可选: 导出 top-N 的最优 pose 为 PDB



可以优化的部分
summary 解析：直接从 CLUSTERING HISTOGRAM 第一行取  
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "ligand_id",
    "dock_success",
    "num_runs",
    "num_clusters",
    "best_run",
    "best_energy",
    "best_cluster_rank",
    "best_cluster_size",
    "cluster_ratio",
    "dlg_path",
    "error",
]


# AutoDock atom type -> 元素符号
_ATOM_TYPE_TO_ELEMENT = {
    "A": "C", "C": "C",
    "N": "N", "NA": "N",
    "O": "O", "OA": "O",
    "S": "S", "SA": "S",
    "H": "H", "HD": "H",
    "F": "F", "P": "P",
    "CL": "Cl", "BR": "Br", "I": "I",
}


def get_ligand_id(dlg_path: Path) -> str:
    """ligand_0013_3I03_protein_pocket.dlg -> ligand_0013"""
    stem = Path(dlg_path).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "ligand":
        return f"{parts[0]}_{parts[1]}"
    return stem


def parse_dlg_summary(dlg_path: Path) -> dict:
    """解析 dlg，返回单个 ligand 的汇总信息。"""
    dlg_path = Path(dlg_path)
    lines = dlg_path.read_text(errors="ignore").splitlines()

    ligand_id = get_ligand_id(dlg_path)
    dock_success = any("Successful Completion" in line for line in lines)

    num_runs = None
    num_clusters = None
    run_results: list[dict] = []
    current_run = None

    for line in lines:
        if "Number of requested GA dockings" in line:
            num_runs = int(line.split("=")[1].split()[0])
        elif "Number of distinct conformational clusters found" in line:
            num_clusters = int(line.split("=")[1].split(",")[0].strip())
        elif line.startswith("DOCKED: USER    Run ="):
            current_run = int(line.split("=")[1].strip())
        elif line.startswith("DOCKED: USER    Estimated Free Energy of Binding"):
            energy = float(line.split("=")[1].split()[0])
            run_results.append({"run": current_run, "energy": energy})

    if not run_results:
        return {
            "ligand_id": ligand_id,
            "dock_success": False,
            "num_runs": num_runs,
            "num_clusters": num_clusters,
            "best_run": None,
            "best_energy": None,
            "best_cluster_rank": None,
            "best_cluster_size": None,
            "cluster_ratio": None,
            "dlg_path": str(dlg_path),
            "error": "no docking pose parsed",
        }

    # AutoDock binding energy 越负越好
    best = min(run_results, key=lambda r: r["energy"])
    best_run = best["run"]
    best_energy = best["energy"]

    best_cluster_rank, best_cluster_size = _parse_cluster_for_run(
        lines, best_run, best_energy,
    )

    cluster_ratio = None
    if best_cluster_size is not None and num_runs:
        cluster_ratio = best_cluster_size / num_runs

    return {
        "ligand_id": ligand_id,
        "dock_success": dock_success,
        "num_runs": num_runs,
        "num_clusters": num_clusters,
        "best_run": best_run,
        "best_energy": best_energy,
        "best_cluster_rank": best_cluster_rank,
        "best_cluster_size": best_cluster_size,
        "cluster_ratio": cluster_ratio,
        "dlg_path": str(dlg_path),
        "error": None,
    }


def _parse_cluster_for_run(
    lines: list[str], target_run: int, target_energy: float,
) -> tuple[int | None, int | None]:
    """
    在 CLUSTERING HISTOGRAM 中找到 target_run 所在的 cluster rank 和大小。
    优先按 run 号匹配；若 histogram 中代表 run 与 target_run 不同（多个 run 能量
    并列时常见），按 lowest binding energy 回退匹配。
    """
    in_cluster_table = False
    rows: list[tuple[int, int, float, int]] = []  # (rank, run, lowest_energy, size)
    for line in lines:
        if "CLUSTERING HISTOGRAM" in line:
            in_cluster_table = True
            continue
        if not in_cluster_table:
            continue
        if "RMSD TABLE" in line:
            break
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        # 表头/横线行 parts[0] 不是数字
        if len(parts) >= 5 and parts[0].isdigit():
            rows.append((
                int(parts[0]),         # rank
                int(parts[2]),         # representative run
                float(parts[1]),       # lowest binding energy
                int(parts[4]),         # cluster size
            ))

    for rank, run, _energy, size in rows:
        if run == target_run:
            return rank, size
    # 回退：histogram 精度到 0.01 kcal/mol
    for rank, _run, energy, size in rows:
        if abs(energy - target_energy) < 0.005:
            return rank, size
    return None, None


def _format_pdb_atom(serial: int, atom_name: str, res_name: str,
                     res_id: int, x: float, y: float, z: float, element: str) -> str:
    return (
        f"HETATM{serial:5d} "
        f"{atom_name:<4s} "
        f"{res_name:>3s} "
        f"X"
        f"{res_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          "
        f"{element:>2s}"
    )


def _guess_element(atom_name: str, atom_type: str | None) -> str:
    if atom_type:
        mapped = _ATOM_TYPE_TO_ELEMENT.get(atom_type.strip().upper())
        if mapped:
            return mapped
    name = atom_name.strip()
    return name[0].upper() if name else "C"


def extract_best_pose_pdb(dlg_path: Path, output_dir: Path,
                          best_run: int, best_energy: float) -> Path:
    """从 dlg 中提取 best_run 对应的 MODEL 块，写为 PDB。"""
    dlg_path = Path(dlg_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ligand_id = get_ligand_id(dlg_path)
    lines = dlg_path.read_text(errors="ignore").splitlines()

    in_target = False
    found = False
    pdb_lines: list[str] = []

    for line in lines:
        if line.startswith("DOCKED: MODEL"):
            model_id = int(line.split()[-1])
            if model_id == best_run:
                in_target = True
                found = True
                pdb_lines.append(f"REMARK ligand_id {ligand_id}")
                pdb_lines.append(f"REMARK best_run {best_run}")
                pdb_lines.append(f"REMARK best_energy {best_energy:.3f} kcal/mol")
            else:
                in_target = False
            continue

        if in_target and line.startswith("DOCKED: ENDMDL"):
            break

        if in_target and line.startswith("DOCKED: ATOM"):
            # 例：DOCKED: ATOM      1  N1  UNL     1      20.729   7.082  29.992 -0.15 -0.36    -0.356 N
            parts = line.split()
            serial = int(parts[2])
            atom_name = parts[3]
            res_name = parts[4]
            res_id = int(parts[5])
            x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
            atom_type = parts[-1]
            element = _guess_element(atom_name, atom_type)
            pdb_lines.append(_format_pdb_atom(
                serial, atom_name, res_name, res_id, x, y, z, element,
            ))

    if not found:
        raise ValueError(f"Cannot find MODEL {best_run} in {dlg_path}")
    if len(pdb_lines) <= 3:
        raise ValueError(f"No atom coordinates extracted from {dlg_path}")

    pdb_lines.append("TER")
    pdb_lines.append("END")

    output_pdb = output_dir / f"{ligand_id}_best.pdb"
    output_pdb.write_text("\n".join(pdb_lines) + "\n")
    return output_pdb


def build_summary_dataframe(dlg_dir: Path) -> pd.DataFrame:
    """扫描 dlg_dir 下所有 dlg，返回汇总 DataFrame。"""
    dlg_dir = Path(dlg_dir)
    dlg_files = sorted(dlg_dir.glob("*.dlg"))
    print(f"Found {len(dlg_files)} dlg files in {dlg_dir}")

    rows = []
    for dlg_path in dlg_files:
        try:
            row = parse_dlg_summary(dlg_path)
        except Exception as e:
            row = {col: None for col in SUMMARY_COLUMNS}
            row.update({
                "ligand_id": get_ligand_id(dlg_path),
                "dock_success": False,
                "dlg_path": str(dlg_path),
                "error": f"{type(e).__name__}: {e}",
            })
        rows.append(row)

    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return df


def format_top_n_text(df_top: pd.DataFrame) -> str:
    """把 top-N DataFrame 渲染成对齐的纯文本。"""
    cols = [
        "ligand_id", "best_energy", "best_run",
        "best_cluster_rank", "best_cluster_size",
        "cluster_ratio", "num_clusters", "num_runs",
    ]
    show = df_top[cols].copy()
    # 数值列格式化
    if "best_energy" in show:
        show["best_energy"] = show["best_energy"].map(
            lambda v: f"{v:.3f}" if pd.notnull(v) else ""
        )
    if "cluster_ratio" in show:
        show["cluster_ratio"] = show["cluster_ratio"].map(
            lambda v: f"{v:.2f}" if pd.notnull(v) else ""
        )
    return show.to_string(index=False)


def write_top_n_text(df: pd.DataFrame, output_txt: Path, top_n: int) -> pd.DataFrame:
    """挑出能量最低的 top_n 个 ligand，打印 + 写 TXT。返回 top_n 子表。"""
    valid = df[df["best_energy"].notnull()].copy()
    df_top = valid.nsmallest(top_n, "best_energy").reset_index(drop=True)

    text = format_top_n_text(df_top)
    header = f"Top {len(df_top)} ligands by best binding energy"
    body = f"{header}\n{'=' * len(header)}\n{text}\n"

    output_txt.write_text(body)
    print()
    print(body)
    return df_top


def export_top_poses(df_top: pd.DataFrame, pose_dir: Path) -> None:
    """导出 top-N ligand 的最优 pose 为 PDB。"""
    pose_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df_top.iterrows():
        if not row.get("dlg_path") or pd.isnull(row.get("best_run")):
            continue
        try:
            pdb_path = extract_best_pose_pdb(
                Path(row["dlg_path"]),
                pose_dir,
                int(row["best_run"]),
                float(row["best_energy"]),
            )
            print(f"Exported: {pdb_path.name}")
        except Exception as e:
            print(f"Failed to export pose for {row['ligand_id']}: {e}", file=sys.stderr)


def analyze_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    top_n: int = 10,
    export_top_poses_flag: bool = False,
) -> pd.DataFrame:
    """主流程：汇总 + top-N 报告 +（可选）导出 pose。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_summary_dataframe(input_dir)

    summary_csv = output_dir / "summary_all_samples.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Generated: {summary_csv}")

    top_txt = output_dir / f"top{top_n}_samples.txt"
    df_top = write_top_n_text(df, top_txt, top_n)
    print(f"Generated: {top_txt}")

    if export_top_poses_flag:
        pose_dir = output_dir / "best_poses"
        print(f"\nExporting top-{top_n} best poses to {pose_dir}/ ...")
        export_top_poses(df_top, pose_dir)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AutoDock batch docking results into a summary DataFrame.",
    )
    parser.add_argument("--input-dir", type=Path, default='/data3/lpw/dock_workdir/autodock/test',
                        help="Directory containing .dlg files.")
    parser.add_argument("--output-dir", type=Path, default='/data3/lpw/dock_workdir/autodock/test'  ,
                        help="Output directory for summary CSV / TXT / poses.")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of best ligands to report (default: 10).")
    parser.add_argument("--export-top-poses", action="store_true",
                        help="Also extract best PDB poses for the top-N ligands.")

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Top N:      {args.top_n}")
    print(f"Export poses: {args.export_top_poses}")
    print()

    analyze_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
        export_top_poses_flag=args.export_top_poses,
    )


if __name__ == "__main__":
    main()
