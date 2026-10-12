import re
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from config import run_pdbq2pdb
from run_mopac.data_utils import parse_pdb, write_df2_pdb, extract_ligand, merge_pocket, select_pocket
from run_mopac.mopac_utils import optimize_h_only
from utils.cov_utils import parse_dlg_energies_multi

'''
氨基酸残基： 平均重原子数：约为 8.2 个。平均总原子数：约为 16.7 个。



'''
# ----------------------------------------------------------------------
# 小工具：加氢 / 删共价原子的 H / 定位口袋共价原子
# ----------------------------------------------------------------------
def _add_h_and_optimize(mol, max_iters=200):
    """加全氢并只优化 H 的位置。返回新 mol（AddHs 非原地）。"""
    mol_h = Chem.AddHs(mol, addCoords=True)
    optimize_h_only(mol_h, max_iters=max_iters)
    return mol_h


def _strip_cov_hydrogen(mol, cov_idx):
    """删除共价原子 cov_idx 上的一个 H，给后续成键腾出价位。返回新 mol。"""
    cov_atom = mol.GetAtomWithIdx(cov_idx)
    h_neighbors = [n for n in cov_atom.GetNeighbors() if n.GetAtomicNum() == 1]
    if not h_neighbors:
        raise ValueError(f"共价原子 idx={cov_idx} 上没有可删除的 H")
    emol = Chem.RWMol(mol)
    emol.RemoveAtom(h_neighbors[0].GetIdx())
    return emol.GetMol()


def _find_pocket_cov_idx(mol, chain_id, res_name, res_seq, atom_name):
    """按 链/残基名/残基号/原子名 在 pocket mol 中定位共价原子，返回其 idx。"""
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        if (
            info.GetChainId().strip() == chain_id
            and info.GetResidueName().strip() == res_name
            and info.GetResidueNumber() == res_seq
            and info.GetName().strip() == atom_name
        ):
            return atom.GetIdx()
    raise ValueError(
        f"未找到共价原子 {chain_id}:{res_name}{res_seq}-{atom_name}"
    )


# ----------------------------------------------------------------------
# 单个 pose 的处理单元（可独立调用，便于单测/并行）
# ----------------------------------------------------------------------
def _prepare_one_pose(i, template_mol, df_rigid_pocket, bond_res, pose_i_dir, mopac_i_dir): # 这是不是不能通过pdb直接转？是不是要assign到mol对象中呢？
    """处理第 i 个 pose：抽配体加氢 + 拼口袋加氢，各删一个共价 H 后写出。"""
    df_pose = parse_pdb(pose_i_dir / f'res_ligand_pose_{i}.pdb')

    # --- ligand：抽纯配体 -> 套模板键级 -> 加氢 -> 删共价原子的 H ---
    df_ligand = extract_ligand(df_pose)
    ligand_pdb = mopac_i_dir / f'pose_ligand_{i}.pdb'
    write_df2_pdb(df_ligand, ligand_pdb)

    ligand_mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=True)        # removeHs=True 只有在 sanitize=True 时才真正生效
    ligand_mol = AllChem.AssignBondOrdersFromTemplate(template_mol, ligand_mol)
    ligand_h = _add_h_and_optimize(ligand_mol)
    ligand_h = _strip_cov_hydrogen(ligand_h, cov_idx=0)                                     #  假设配体共价原子是 idx 0，需核实
    Chem.MolToPDBFile(ligand_h, str(mopac_i_dir / f'ligand_addh_{i}.pdb'))

    # --- pocket：flex(来自 pose) + rigid 拼接 -> 裁到配体3Å整残基 -> 加氢 -> 删共价残基的 H ---


    df_pocket = merge_pocket(df_pose, df_rigid_pocket)
    # 加氢前把对接用的大口袋裁到参考配体(template)周围 3Å 的整残基。
    # 用 template 而非各 pose 自身的配体坐标，保证同一 pdb_id 的所有 pose 得到
    # 完全一致的口袋（残基/原子数相同），MOPAC 总能量才可跨 pose 比较。
    # template 与 pose 同坐标系且共价点重合，共价残基必被包含。
    ref_coor = template_mol.GetConformer().GetPositions()
    # import ipdb; ipdb.set_trace()
    df_pocket = select_pocket(df_pocket, ref_coor, 5.0, any_atom=True)
    pocket_pdb = mopac_i_dir / f'pose_pocket_{i}.pdb'
    write_df2_pdb(df_pocket, pocket_pdb)

    pocket_mol = Chem.MolFromPDBFile(str(pocket_pdb), removeHs=False)
    pocket_h = _add_h_and_optimize(pocket_mol)
    cov_idx = _find_pocket_cov_idx(pocket_h, *bond_res)
    pocket_h = _strip_cov_hydrogen(pocket_h, cov_idx)
    Chem.MolToPDBFile(pocket_h, str(mopac_i_dir / f'pocket_addh_{i}.pdb'))


# ----------------------------------------------------------------------
# single：处理一个 pdb_id 的全部 pose
# ----------------------------------------------------------------------
def single_preparation(pdb_id, bond_res, pose_dir, data_dir, mopac_dir, result_dir, sample_nums=10):
    pose_i_dir = pose_dir / pdb_id
    data_i_dir = data_dir / pdb_id
    mopac_i_dir = mopac_dir / pdb_id
    result_i_dir = result_dir / pdb_id / 'conf_1'       # 这里进行了一点修改
    mopac_i_dir.mkdir(parents=True, exist_ok=True)

    # bond_res 形如 'D:SER-195-OG'
    chain_id, res = bond_res.split(':')
    res_name, res_seq, atom_name = res.split('-')
    bond_res = (chain_id, res_name, int(res_seq), atom_name)

    # per-pdb：刚性口袋 pdbqt -> pdb 只转一次
    rigid_pdb = mopac_i_dir / 'pocket_rigid.pdb'
    run_pdbq2pdb(
        ['-f', str(result_i_dir / 'pocket_rigid.pdbqt'), '-o', str(rigid_pdb)],
        cwd=mopac_i_dir,
    )
    df_rigid_pocket = parse_pdb(rigid_pdb)

    # per-pdb：模板分子只读一次（用于配体键级还原，必须 removeHs=True）
    template_mol = Chem.SDMolSupplier(
        str(data_i_dir / f'{pdb_id}_ligand.sdf'), removeHs=True
    )[0]

    # pose 编号 = DLG run 号（get_pose 跨 conf 重编号后连续），本身并不按结合能排序。
    # 要用“docking 打分最好的前 sample_nums 个 pose”做 mopac，必须先按结合能（越负越好）排序再截取。
    def _pose_idx(p):
        return int(re.search(r'_(\d+)\.pdb$', p.name).group(1))

    pose_files = list(pose_i_dir.glob('res_ligand_pose_*.pdb'))
    assert pose_files, f"未找到任何 pose 文件: {pose_i_dir}"
    energies = parse_dlg_energies_multi(result_dir / pdb_id)     # {pose_idx: energy}，编号与 res_ligand_pose_{i} 一致
    # 按结合能升序（越负越好在前）；缺失能量的 pose 用 inf 兜底并按编号稳定排序，退化为旧的取前 N 行为
    pose_files.sort(key=lambda p: (energies.get(_pose_idx(p), float('inf')), _pose_idx(p)))
    if sample_nums is not None:
        pose_files = pose_files[:sample_nums]

    # per-pose：串行循环（要给单 pdb 提速时，把这层换成 joblib.Parallel 即可）
    for pose_file in pose_files:
        _prepare_one_pose(_pose_idx(pose_file), template_mol, df_rigid_pocket, bond_res, pose_i_dir, mopac_i_dir)


if __name__ == "__main__":
    pdb_id = '1RWN'
    sample_nums = 100
    # 注意：以下是“类别父目录”，single 内部会自动拼 /pdb_id
    pose_dir = Path('/data3/lpw/dock_workdir/autodock_cov/pose')
    data_dir = Path('/data3/lpw/data/covdocker/data/processed/bonded')
    mopac_dir = Path('/data3/lpw/dock_workdir/autodock_cov/mopac5')
    result_dir = Path('/data3/lpw/dock_workdir/autodock_cov/result6')
    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')

    df_info = pd.read_csv(info_file)
    bond = df_info.loc[df_info['pdb_id'] == pdb_id, 'bond'].values[0]   # 'OG SER D 195-C2 APA D 301'
    name, res_name, chain_id, res_seq = bond.split('-')[0].split(' ')
    bond_res = f'{chain_id}:{res_name}-{res_seq}-{name}'                 # 'D:SER-195-OG'

    single_preparation(pdb_id, bond_res, pose_dir, data_dir, mopac_dir, result_dir, sample_nums)
