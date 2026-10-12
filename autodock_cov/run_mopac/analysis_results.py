from collections.abc import Iterable
from pathlib import Path
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from traceback import format_exc


'''
不能按照autodock的pose去划分3A口袋，应该按照原始内容划分3A口袋，否则不会有orcale的提升

oracle 
autodock是可以做到 40%
pose_moapc 可以做到46%   TOP-1 14.29%




按照原始pocket划分3A口袋
['2EX8', '2G8E', '2HWP', '2OP3', '3BEB', '3C9W', '5DP7', '5HZE', '5IYT']
mopac Oracle    Success Rate: 22.22%
mopac Top-1     Success Rate: 0.00%
autodock Oracle Success Rate: 11.11%

按照pose划分3A口袋
['2EX8', '2G8E', '2HWP', '2OP3', '3BEB', '3C9W', '5DP7', '5HZE', '5IYT']
mopac Oracle    Success Rate: 11.11%
mopac Top-1     Success Rate: 0.00%
autodock Oracle Success Rate: 11.11%



挑选14个样本是orcalesuccess但是
mopac Oracle    Success Rate: 14.29%
mopac Top10     Success Rate: 7.14%
mopac Top-1     Success Rate: 7.14%
autodock Oracle Success Rate: 0.00%
autodock Top-1 Success Rate: 0.00%


使用70个样本，但是有11个样本电荷错误，剩下59个样本
Total PDBs analyzed: 59
mopac Oracle    Success Rate: 54.24%
mopac Top10     Success Rate: 49.15%
mopac Top-1     Success Rate: 40.68%
autodock Oracle Success Rate: 76.27%
autodock Top-1 Success Rate: 30.51%





'''




def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """两组同序坐标的直接 RMSD（不做叠合）。"""
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def _pose_rmsd(pose_path, ligand_gt_mol, gt_coords):
    """单个 MOPAC pose 与晶体配体的重原子直接 RMSD（含对称等价取最小）。

    每个 pose 都**独立**建立 gt<->pose 的原子映射：MOPAC 输出里不同 pose 的
    重原子写出顺序并不一致，因此绝不能复用其它 pose 的映射。无法 sanitize /
    键感知失败 / 匹配不上（畸变 pose）时返回 None，交由上层记为 NaN。
    """
    # 带 sanitize 读取（默认），用于键感知与子结构匹配；removeHs 后即为重原子子图，
    # 其构象坐标与 GetSubstructMatches 返回的原子下标一一对应。
    mol = Chem.MolFromPDBFile(str(pose_path), removeHs=True)
    if mol is None:                                              # 畸变 pose，价态校验失败
        return None
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(ligand_gt_mol, mol)
    except Exception:
        return None
    matches = mol.GetSubstructMatches(ligand_gt_mol, uniquify=False)
    if not matches:
        return None
    coords = mol.GetConformer().GetPositions()                  # 该 pose 自身重原子顺序
    if coords.shape[0] != gt_coords.shape[0]:
        return None
    # matches[k] 把 gt 第 k 个原子映射到本 pose 的原子下标 → coords[list(mt)] 与 gt 同序
    return min(_rmsd(gt_coords, coords[list(mt)]) for mt in matches)


def get_rmsd(pdb_id, data_dir, mopac_pose_dir, dock_nums):
    """计算单个 PDB 的全部 MOPAC pose 与晶体配体之间的 RMSD。

    每个 pose 单独建立原子映射（见 _pose_rmsd）。无法建立映射的畸变 pose
    记为 NaN，从而保持 pose_idx 与能量/轨迹文件的一一对应。
    """
    data_i_dir = Path(data_dir) / pdb_id
    pose_i_dir = Path(mopac_pose_dir) / pdb_id
    assert pose_i_dir.exists(), f"Pose directory does not exist: {pose_i_dir}"

    ligand_gt_file = data_i_dir / f"{pdb_id}_ligand.sdf"
    ligand_gt_mol = Chem.SDMolSupplier(str(ligand_gt_file))[0]
    assert ligand_gt_mol is not None, f"Failed to read ligand: {ligand_gt_file}"
    ligand_gt_mol = Chem.RemoveHs(ligand_gt_mol)
    gt_coords = ligand_gt_mol.GetConformer().GetPositions()      # (n_heavy, 3)

    def pose_file(i):
        return pose_i_dir / f"mopac_ligand_{i}.pdb"

    pose_files = sorted(
        pose_i_dir.glob("mopac_ligand_*.pdb"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    pose_indices = [int(p.stem.rsplit("_", 1)[1]) for p in pose_files]
    assert pose_indices, f"No mopac ligand files found: {pose_i_dir}"

    rows, n_fail = [], 0
    for pose_idx in pose_indices:
        rmsd_value = _pose_rmsd(pose_file(pose_idx), ligand_gt_mol, gt_coords)
        if rmsd_value is None:                                   # 畸变 pose，记 NaN
            n_fail += 1
            rmsd_value = float("nan")
        rows.append({"pdb_id": pdb_id, "pose_idx": pose_idx, "rmsd": rmsd_value})

    if n_fail:
        print(
            f"[{pdb_id}] {n_fail}/{len(pose_indices)} 个 pose 无法建立原子映射，"
            f"RMSD 记为 NaN"
        )

    return pd.DataFrame(rows)


def rmsd_sort_by_energy(pdb_id, data_dir, mopac_pose_dir, traj_dir, dock_nums):
    """计算单个 PDB 的 RMSD 和最终能量，并按能量从低到高保存。"""
    
    try:
        df_rmsd = get_rmsd(pdb_id, data_dir, mopac_pose_dir, dock_nums)
    except:
        print(f"Error calculating RMSD for {pdb_id}: {format_exc()}")
        return None
    
    traj_i_dir = Path(traj_dir) / pdb_id
    energies = []

    for pose_idx in df_rmsd["pose_idx"]:
        traj_file = traj_i_dir / f"complex_{pose_idx}.npz"
        assert traj_file.exists(), f"Missing trajectory file: {traj_file}"

        with np.load(traj_file) as traj:
            energies.append(float(traj["energy"][-1]))

    df_rmsd["energy"] = energies
    df_rmsd = df_rmsd.sort_values("energy").reset_index(drop=True)

    pose_i_dir = Path(mopac_pose_dir) / pdb_id
    output_file = pose_i_dir / f"{pdb_id}_mopac_rmsd.csv"
    df_rmsd.to_csv(output_file, index=False)
    return df_rmsd


def batch_rmsd_analysis(
    pdb_ids, data_dir, mopac_pose_dir, traj_dir, dock_nums
) -> None:
    """依次计算多个 PDB 的 MOPAC RMSD 和能量。"""
    for pdb_id in pdb_ids:
        rmsd_sort_by_energy(
            pdb_id, data_dir, mopac_pose_dir, traj_dir, dock_nums
        )


def total_analysis(pdb_ids: Iterable[str], mopac_pose_dir, autodock_pose_dir) -> pd.DataFrame:
    """汇总每个 PDB 的 oracle / top-1 RMSD（mopac vs autodock）。"""
    mopac_pose_dir = Path(mopac_pose_dir)
    autodock_pose_dir = Path(autodock_pose_dir)
    rows = []

    for pdb_id in pdb_ids:
        mopac_rmsd_file = mopac_pose_dir / pdb_id / f"{pdb_id}_mopac_rmsd.csv"
        autodock_rmsd_file = autodock_pose_dir / pdb_id / f"{pdb_id}_docking_rmsd.csv"
        if not mopac_rmsd_file.exists():
            print(f"mopac RMSD file does not exist: {mopac_rmsd_file}")
            continue
        if not autodock_rmsd_file.exists():
            print(f"autodock RMSD file does not exist: {autodock_rmsd_file}")
            continue

        df_mopac_rmsd = pd.read_csv(mopac_rmsd_file)
        df_autodock_rmsd = pd.read_csv(autodock_rmsd_file)

        # 一个 pdb 只 append 一行，mopac / autodock 字段对齐在同一行
        df_mopac_sorted = df_mopac_rmsd.sort_values("energy")                        # 按能量升序，用于取 top-k
        rows.append(
            {
                "pdb_id": pdb_id,
                "mopac_best_rmsd": df_mopac_rmsd["rmsd"].min(),                       # mopac oracle：全部 pose 最小
                "mopac_top10_rmsd": df_mopac_sorted["rmsd"].head(10).min(),           # mopac top-10：能量前10里最小
                "mopac_top_rmsd": df_mopac_sorted["rmsd"].iloc[0],                    # mopac top-1：能量最低那个
                "autodock_best_rmsd": df_autodock_rmsd["rmsd"].min(),                 # autodock oracle
                "autodock_top_rmsd": df_autodock_rmsd["rmsd"].iloc[0],
            }
        )

    df_results = pd.DataFrame(rows)
    if not df_results.empty:
        mopac_oracle_success_rate = (df_results["mopac_best_rmsd"] <= 2.0).mean()
        mopac_top10_success_rate = (df_results["mopac_top10_rmsd"] <= 2.0).mean()
        mopac_top1_success_rate = (df_results["mopac_top_rmsd"] <= 2.0).mean()
        autodock_oracle_success_rate = (df_results["autodock_best_rmsd"] <= 2.0).mean()
        autodock_top1_success_rate = (df_results["autodock_top_rmsd"] <= 2.0).mean()
        
        print(f"Total PDBs analyzed: {len(df_results)}")
        print(f"mopac Oracle    Success Rate: {mopac_oracle_success_rate:.2%}")
        print(f"mopac Top10     Success Rate: {mopac_top10_success_rate:.2%}")
        print(f"mopac Top-1     Success Rate: {mopac_top1_success_rate:.2%}")
        print(f"autodock Oracle Success Rate: {autodock_oracle_success_rate:.2%}")
        print(f'autodock Top-1 Success Rate: {autodock_top1_success_rate:.2%}')
    df_results.to_csv(mopac_pose_dir.parent / "mopac_pose5_analysis.csv", index=False)
    return df_results


if __name__ == "__main__":
    data_dir = Path("/data3/lpw/data/covdocker/data/processed/bonded")
    autodock_pose_dir = Path("/data3/lpw/dock_workdir/autodock_cov/pose6")
    mopac_pose_dir = Path("/data3/lpw/dock_workdir/autodock_cov/mopac_pose5")
    traj_dir = Path("/data3/lpw/dock_workdir/autodock_cov/mopac_traj5")
    info_file = Path("/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv")
    dock_nums = 100
    pdb_ids = []                                    # 这里得到的应当是成功跑完mopac的pdb_id
    for pdb_id in sorted(os.listdir(mopac_pose_dir)):
        if (mopac_pose_dir / pdb_id).is_dir():
            pdb_ids.append(pdb_id)
    # pdb_ids = ['2EX8', '2G8E', '2HWP', '2OP3', '3BEB', '3C9W', '5DP7', '5HZE', '5IYT']

    # import ipdb; ipdb.set_trace()
    batch_rmsd_analysis(
        pdb_ids, data_dir, mopac_pose_dir, traj_dir, dock_nums=dock_nums
    )
    total_analysis(pdb_ids, mopac_pose_dir, autodock_pose_dir)
