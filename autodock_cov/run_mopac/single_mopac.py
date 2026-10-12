import re
import subprocess
from pathlib import Path
from shutil import copyfile
from rdkit import Chem
import pandas as pd
from run_mopac.mopac_utils import elem_coor_to_xyz, extract_traj, assign_coor_to_mol
import numpy as np



'''
能量项
E_complex ≈ E_ligand(自身构象能) + E_pocket(基本固定) + E_相互作用(ligand↔pocket)
如果ligand直接逃脱口袋了要怎么办呢？？

'''


# 与现有 complex_mopout.mop 一致的默认关键词（运行时再追加 AUX）                                  # 这个字段要怎么学习一下啊。。。。(有点复杂)
DEFAULT_MOPAC_KW = "PM7 EPS=78.4 CHARGE=0 MOZYME THREADS=1 LBFGS T=1.0D PRNT=2"    # "PM7 THREADS=1 EPS=78.4 CHARGE=0 T=1.0D PRNT=2"  原本的参数设置



def _mol_to_mopac_block(mol, opt_flag=False, only_opt_H=False):
    """rdkit mol -> MOPAC xyz 文本块（带优化标志位）。"""
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions()
    return elem_coor_to_xyz(elements, coords, only_opt_H=only_opt_H, opt_flag=opt_flag)


def _write_mop(mop_file, mopac_kw, complex_block, comment="complex run"):
    with open(mop_file, "w") as f:
        f.write(f"{mopac_kw}\n")
        f.write(f"# {comment}\n\n")
        f.write(complex_block)
        f.write("\n")


def build_complex_mop(pdb_id, mopac_dir, out_i_dir, idx, mopac_kw=DEFAULT_MOPAC_KW):
    """读取 mopac_dir/pdb_id 下某 pose 的 ligand_addh/pocket_addh，拼 MOPAC 输入写到 out_i_dir。"""
    mopac_i_dir = Path(mopac_dir) / pdb_id
    lig_file = mopac_i_dir / f"ligand_addh_{idx}.pdb"
    pkt_file = mopac_i_dir / f"pocket_addh_{idx}.pdb"

    lig_mol = Chem.MolFromPDBFile(str(lig_file), removeHs=False)
    pkt_mol = Chem.MolFromPDBFile(str(pkt_file), removeHs=False)
    if lig_mol is None or pkt_mol is None:
        raise ValueError(f"[{pdb_id}] pose {idx} 读取 addh pdb 失败")

    lig_block = _mol_to_mopac_block(lig_mol, opt_flag=True)    # 配体全优化
    pkt_block = _mol_to_mopac_block(pkt_mol, opt_flag=False)   # 口袋冻结
    complex_block = lig_block + "\n" + pkt_block
    # import ipdb; ipdb.set_trace()
    mop_file = Path(out_i_dir) / f"complex_{idx}.mop"
    _write_mop(mop_file, mopac_kw + " AUX", complex_block, comment=f"{pdb_id} pose {idx}")
    return mop_file


def single_mopac(pdb_id, mopac_dir, result_dir,
                 sample_nums, pose_nums, mopac_kw=DEFAULT_MOPAC_KW, run=True):
    """对一个 pdb 的全部已准备 pose 生成 MOPAC 输入并（可选）运行；输出写到 result_dir/pdb_id。"""
    mopac_i_dir = Path(mopac_dir) / pdb_id
    out_i_dir = Path(result_dir) / pdb_id
    out_i_dir.mkdir(parents=True, exist_ok=True)

    lig_files = sorted(mopac_i_dir.glob("ligand_addh_*.pdb"))
    assert len(lig_files) == pose_nums, f"[{pdb_id}] 采样出现问题，采样数 {len(lig_files)} != {pose_nums}"
    if sample_nums is not None:
        lig_files = lig_files[:sample_nums]

    for lig_file in lig_files:
        idx = int(lig_file.stem.rsplit("_", 1)[1])
        mop_file = build_complex_mop(
            pdb_id, mopac_dir, out_i_dir, idx,
            mopac_kw=mopac_kw,
        )
        if run:
            # mopac 把 complex_{idx}.out/.aux/.arc 等写到 mop 同目录(out_i_dir)
            subprocess.run(["mopac", str(mop_file)], cwd=str(out_i_dir))


def extract_mopac_pose(
    pdb_id, idx, mopac_dir, result_dir,
    pose_dir, traj_dir,
    mopac_kw=DEFAULT_MOPAC_KW,
):
    """提取 MOPAC 最终优化构象，并保存配体 PDB 和完整轨迹。"""
    aux_file = Path(result_dir) / pdb_id / f"complex_{idx}.aux"
    lig_file = Path(mopac_dir) / pdb_id / f"ligand_addh_{idx}.pdb"

    pose_path = Path(pose_dir) / pdb_id / f"mopac_ligand_{idx}.pdb"
    traj_path = Path(traj_dir) / pdb_id / f"complex_{idx}.npz"

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    traj = extract_traj(mopac_kw, aux_file)
    np.savez_compressed(traj_path, **traj)

    lig_mol = Chem.MolFromPDBFile(str(lig_file), removeHs=False)
    if lig_mol is None:
        raise ValueError(f"[{pdb_id}] pose {idx} 读取 ligand_addh 失败")

    n_lig = lig_mol.GetNumAtoms()
    final_coor = traj["coor"][-1]

    if len(final_coor) < n_lig:
        raise ValueError(
            f"[{pdb_id}] pose {idx} aux 原子数 "
            f"{len(final_coor)} < 配体 {n_lig}"
        )

    lig_mol = assign_coor_to_mol(lig_mol, final_coor[:n_lig])
    Chem.MolToPDBFile(lig_mol, str(pose_path))

    return pose_path

def single_extract_pose(pdb_id, mopac_dir, result_dir,
                        mopac_pose_dir, mopac_traj_dir, mopac_kw=DEFAULT_MOPAC_KW):
    """对一个 pdb 已跑完的所有 complex_{idx}.aux 批量提取配体 pose。

    调用前需先跑 prepare_pose_dirs：建目录并拷贝 gt_ligand / 10A pocket / bond_res。
    """
    result_i_dir = Path(result_dir) / pdb_id
    aux_files = sorted(
        result_i_dir.glob("complex_*.aux"),
        key=lambda p: int(re.search(r"_(\d+)\.aux$", p.name).group(1)),
    )
    outs = []
    for aux in aux_files:
        idx = int(re.search(r"_(\d+)\.aux$", aux.name).group(1))
        outs.append(extract_mopac_pose(pdb_id, idx, mopac_dir, result_dir, mopac_pose_dir, mopac_traj_dir, mopac_kw))
    return outs


# ----------------------------------------------------------------------
# pose 级并行用的小工具（供 batch_mopac 把 (pdb, pose) 铺平成独立任务）
# ----------------------------------------------------------------------
def list_pose_indices(pdb_id, mopac_dir, pose_nums, sample_nums=None):
    """列出某 pdb 已准备好的 pose 序号（带数量校验与采样裁剪）。"""
    mopac_i_dir = Path(mopac_dir) / pdb_id
    lig_files = sorted(
        mopac_i_dir.glob("ligand_addh_*.pdb"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    assert len(lig_files) == pose_nums, f"[{pdb_id}] 采样出现问题，采样数 {len(lig_files)} != {pose_nums}"
    if sample_nums is not None:
        lig_files = lig_files[:sample_nums]
    return [int(p.stem.rsplit("_", 1)[1]) for p in lig_files]


def prepare_pose_dirs(pdb_id, bond_res, data_dir, result_dir, mopac_pose_dir, mopac_traj_dir):
    """每个 pdb 跑 pose 前的一次性准备：建目录 + 拷贝 gt_ligand / 10A pocket / bond_res。"""
    (Path(result_dir) / pdb_id).mkdir(parents=True, exist_ok=True)
    pose_id_dir = Path(mopac_pose_dir) / pdb_id
    traj_id_dir = Path(mopac_traj_dir) / pdb_id
    pose_id_dir.mkdir(parents=True, exist_ok=True)
    traj_id_dir.mkdir(parents=True, exist_ok=True)

    data_id_dir = Path(data_dir) / pdb_id
    copyfile(data_id_dir / f"{pdb_id}_ligand.sdf", pose_id_dir / f"{pdb_id}_ligand.sdf")
    copyfile(data_id_dir / f"{pdb_id}_10Apocket.pdb", pose_id_dir / f"{pdb_id}_10Apocket.pdb")
    with open(pose_id_dir / f"{pdb_id}_bond_res.txt", "w") as f:
        f.write(bond_res)


def run_one_pose(pdb_id, idx, mopac_dir, result_dir, mopac_pose_dir, mopac_traj_dir,
                 mopac_kw=DEFAULT_MOPAC_KW, run=True):
    """单个 pose：拼 .mop -> 跑 MOPAC -> 提取配体 pose/轨迹。pose 级并行的最小单元。

    同一 pdb 的不同 pose 写各自的 complex_{idx}.*，文件名不冲突，可安全并行。
    """
    out_i_dir = Path(result_dir) / pdb_id
    mop_file = build_complex_mop(pdb_id, mopac_dir, out_i_dir, idx, mopac_kw=mopac_kw)
    if run:
        subprocess.run(["mopac", str(mop_file)], cwd=str(out_i_dir))
        extract_mopac_pose(pdb_id, idx, mopac_dir, result_dir, mopac_pose_dir, mopac_traj_dir, mopac_kw)
    return idx


if __name__ == "__main__":
    mopac_dir = Path("/data3/lpw/dock_workdir/autodock_cov/mopac5")
    result_dir = Path("/data3/lpw/dock_workdir/autodock_cov/mopac_result5")
    mopac_pose_dir = Path("/data3/lpw/dock_workdir/autodock_cov/mopac_pose5")
    mopac_traj_dir = Path("/data3/lpw/dock_workdir/autodock_cov/mopac_traj5")
    data_dir = Path("/data3/lpw/data/covdocker/data/processed/bonded")
    info_file = Path("/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv")
    df_info = pd.read_csv(info_file)

    

    pdb_id = "1AYW"
    bond = df_info[df_info['pdb_id'] == pdb_id]['bond'].values[0]  # 'OG SER D 195-C2 APA D 301'
    name, res_name, chain_id, res_seq = bond.split('-')[0].split()
    bond_res = f"{chain_id}:{res_name}-{res_seq}-{name}"  # 'D:SER-195-OG'
    pose_nums = 100  # docking中实际生成的pose数量（理论上一定要满足的）
    sample_nums = 1   # 用于进行debug，只跑一个来测试
    single_mopac(pdb_id, mopac_dir, result_dir=result_dir, sample_nums=sample_nums, pose_nums=pose_nums)
    prepare_pose_dirs(pdb_id, bond_res, data_dir, result_dir, mopac_pose_dir, mopac_traj_dir)
    single_extract_pose(pdb_id, mopac_dir, result_dir, mopac_pose_dir, mopac_traj_dir)
