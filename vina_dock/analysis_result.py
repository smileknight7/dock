#!/usr/bin/env python3
"""
AutoDock Vina 批量对接结果分析。

- 解析每个 ligand 的 *_out.pdbqt，汇总成一张 DataFrame
- 输出 summary CSV（所有 ligand）+ top-N TXT（分数最低的 N 个）
- 可选: 导出 top-N 的最优 pose（MODEL 1）为 PDB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "ligand_id",
    "dock_success",
    "best_score",
    "worst_score",
    "score_range",
    "pose_count",
    "pdbqt_path",
    "error",
]


# Vina/AutoDock atom type -> 元素符号
_ATOM_TYPE_TO_ELEMENT = {
    "A": "C", "C": "C",
    "N": "N", "NA": "N",
    "O": "O", "OA": "O",
    "S": "S", "SA": "S",
    "H": "H", "HD": "H",
    "F": "F", "P": "P",
    "CL": "Cl", "BR": "Br", "I": "I",
}


def get_ligand_id(pdbqt_path: Path) -> str:
    """ligand_0025_out.pdbqt -> ligand_0025"""
    stem = Path(pdbqt_path).stem
    if stem.endswith("_out"):
        stem = stem[: -len("_out")]
    return stem


def parse_vina_pdbqt(pdbqt_path: Path) -> dict:
    """解析 vina *_out.pdbqt，返回单个 ligand 的汇总信息。"""
    pdbqt_path = Path(pdbqt_path)
    ligand_id = get_ligand_id(pdbqt_path)
    lines = pdbqt_path.read_text(errors="ignore").splitlines()

    scores: list[float] = []
    for line in lines:
        # 例：REMARK VINA RESULT:    -6.066      0.000      0.000
        if line.startswith("REMARK VINA RESULT:"):
            parts = line.split()
            try:
                scores.append(float(parts[3]))
            except (IndexError, ValueError):
                continue

    if not scores:
        return {
            "ligand_id": ligand_id,
            "dock_success": False,
            "best_score": None,
            "worst_score": None,
            "score_range": None,
            "pose_count": 0,
            "pdbqt_path": str(pdbqt_path),
            "error": "no VINA RESULT remark parsed",
        }

    best = min(scores)
    worst = max(scores)
    return {
        "ligand_id": ligand_id,
        "dock_success": True,
        "best_score": best,
        "worst_score": worst,
        "score_range": worst - best,
        "pose_count": len(scores),
        "pdbqt_path": str(pdbqt_path),
        "error": None,
    }


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


def extract_best_pose_pdb(pdbqt_path: Path, output_dir: Path,
                          best_score: float) -> Path:
    """从 *_out.pdbqt 中提取 MODEL 1（vina 已按分数排序），写为 PDB。"""
    pdbqt_path = Path(pdbqt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ligand_id = get_ligand_id(pdbqt_path)
    lines = pdbqt_path.read_text(errors="ignore").splitlines()

    in_first_model = False
    seen_model = False
    pdb_lines: list[str] = [
        f"REMARK ligand_id {ligand_id}",
        f"REMARK best_score {best_score:.3f} kcal/mol",
    ]

    for line in lines:
        if line.startswith("MODEL"):
            if seen_model:
                break  # 只取第一个 MODEL
            seen_model = True
            in_first_model = True
            continue

        if not in_first_model:
            continue

        if line.startswith("ENDMDL"):
            break

        # 跳过 PDBQT 特有的拓扑标记
        if line.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF", "REMARK")):
            continue

        if line.startswith(("ATOM", "HETATM")):
            # PDBQT 例：ATOM      1  C   UNL     1      12.403  35.779  18.292  1.00  0.00     0.140 C
            parts = line.split()
            try:
                serial = int(parts[1])
                atom_name = parts[2]
                res_name = parts[3]
                res_id = int(parts[4])
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                atom_type = parts[-1]
            except (IndexError, ValueError):
                continue
            element = _guess_element(atom_name, atom_type)
            pdb_lines.append(_format_pdb_atom(
                serial, atom_name, res_name, res_id, x, y, z, element,
            ))

    if len(pdb_lines) <= 2:
        raise ValueError(f"No atom coordinates extracted from {pdbqt_path}")

    pdb_lines.append("TER")
    pdb_lines.append("END")

    output_pdb = output_dir / f"{ligand_id}_best.pdb"
    output_pdb.write_text("\n".join(pdb_lines) + "\n")
    return output_pdb


def build_summary_dataframe(input_dir: Path) -> pd.DataFrame:
    """递归扫描 input_dir 下所有 *_out.pdbqt，返回汇总 DataFrame。"""
    input_dir = Path(input_dir)
    pdbqt_files = sorted(input_dir.rglob("*_out.pdbqt"))
    print(f"Found {len(pdbqt_files)} vina output files in {input_dir}")

    rows = []
    for path in pdbqt_files:
        try:
            row = parse_vina_pdbqt(path)
        except Exception as e:
            row = {col: None for col in SUMMARY_COLUMNS}
            row.update({
                "ligand_id": get_ligand_id(path),
                "dock_success": False,
                "pdbqt_path": str(path),
                "error": f"{type(e).__name__}: {e}",
            })
        rows.append(row)

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def format_top_n_text(df_top: pd.DataFrame) -> str:
    cols = ["ligand_id", "best_score", "worst_score", "score_range", "pose_count"]
    show = df_top[cols].copy()
    for c in ("best_score", "worst_score", "score_range"):
        show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notnull(v) else "")
    return show.to_string(index=False)


def write_top_n_text(df: pd.DataFrame, output_txt: Path, top_n: int) -> pd.DataFrame:
    valid = df[df["best_score"].notnull()].copy()
    df_top = valid.nsmallest(top_n, "best_score").reset_index(drop=True)

    text = format_top_n_text(df_top)
    header = f"Top {len(df_top)} ligands by best Vina score"
    body = f"{header}\n{'=' * len(header)}\n{text}\n"

    output_txt.write_text(body)
    print()
    print(body)
    return df_top


def export_top_poses(df_top: pd.DataFrame, pose_dir: Path) -> None:
    pose_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df_top.iterrows():
        if not row.get("pdbqt_path") or pd.isnull(row.get("best_score")):
            continue
        try:
            pdb_path = extract_best_pose_pdb(
                Path(row["pdbqt_path"]),
                pose_dir,
                float(row["best_score"]),
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
        description="Analyze Vina batch docking results into a summary DataFrame.",
    )
    parser.add_argument("--input-dir", type=Path, default='/data3/lpw/dock_workdir/autodock/test',
                        help="Directory containing vina output (recursively scans *_out.pdbqt).")
    parser.add_argument("--output-dir", type=Path, default='/data3/lpw/dock_workdir/autodock/test',
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
