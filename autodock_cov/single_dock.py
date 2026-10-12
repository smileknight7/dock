from pathlib import Path
from config import run_prepare_altconfs, run_prepare_covalent, run_prepare_receptor4, run_prepare_ligand4, run_prepare_flex_receptor4, run_prepare_dpf4, run_prepare_gpf4,run_autogrid4, run_autodock4
from utils.cov_utils import replace_file_content
import pandas as pd
from openbabel import pybel
import shutil
import subprocess
import os

'''
应该还要从对接结果中加上采样构象最丰富的5个pose
covalent.py脚本是有点问题的，需要对脚本进行一点修改
但是当前这个好像只能测试redock
'''


def _dock_one_conformer(conf_sdf, sub, bond_res, data_i_dir, ligand_indices,
                        grid_center, grid_npts, ga_run):
    '''对单个构象做共价对接:graft → split → gpf/dpf → autogrid → autodock,产物写到 sub 目录。
    完整保留原有 prepare_covalent / prepare_flexreceptor4 流程,只是工作目录换成 sub、配体换成 conf_sdf、ga_run 改为传入值。'''
    os.makedirs(sub, exist_ok=True)

    # 第一步align cov_ligand
    run_prepare_covalent(['--ligand', f'{conf_sdf}',
                                '--ligindices', ligand_indices,
                                '--receptor', f'{data_i_dir}/pocket.pdb',
                                '--residue', f'{bond_res}',
                                '--outputfile', f'{sub}/ligand_covalent.pdb'],
                                cwd=sub)

    # 第二步 构建ligand和pocket的pdbqt文件
    run_prepare_receptor4(['-r', f'{data_i_dir}/pocket.pdb', '-o', f'{sub}/pocket.pdbqt'], cwd=sub)
    run_prepare_receptor4(['-r', f'{sub}/ligand_covalent.pdb'], cwd=sub)

    # 第三步 构建flex部分
    run_prepare_flex_receptor4(['-r', f'{sub}/pocket.pdbqt', '-s', f'pocket:{bond_res}'], cwd=sub)
    run_prepare_flex_receptor4(['-r', f'{sub}/ligand_covalent.pdbqt', '-s', f'ligand_covalent:{bond_res}'], cwd=sub)

    # 第四步构建gpf，和dpf参数文件
    run_prepare_gpf4(['-r', 'pocket_rigid.pdbqt',                   # gpf 是计算格点的参数文件(Grid Parameter File)
                            '-l', 'ligand_covalent_flex.pdbqt',
                            '-p', f'gridcenter={grid_center}',
                            '-p', f'npts={grid_npts}',
                            '-o', 'autogrid_config.gpf'], cwd=sub)

    run_prepare_dpf4(['-r', 'pocket_rigid.pdbqt',                  # dpf 是要提供进行docking的参数文件(Docking Parameter File)
                        '-x', 'ligand_covalent_flex.pdbqt',             # ga_pop_size   种群数量
                        '-l',  'ligand_covalent_flex.pdbqt',
                        '-o', 'autodock_config.dpf',
                        '-p', 'move=empty',
                        '-p', f'ga_run={ga_run}',                       # 总次数 / 构象数(3 构象时=100)
                        '-p', 'ga_num_evals=2500000',                   # 默认就是这个本来想放大的
                        '-p', 'ga_mutation_rate=0.03',                  # 默认0.02
                        '-p','ga_pop_size=150'], cwd=sub)               # 默认150   种群个体数

    # 第五步进行对接
    (sub / 'empty').touch()          #  dock the covalent ligand as a flexible residue and ignore any 'true' ligand ("move='empty'")
    # manually edited to set the appropriate energy model so that the docking score corresponds to the interaction between the flexible residue (the ligand) and the rigid receptor
    replace_file_content(sub/'autogrid_config.gpf', 'unbound_model bound', 'unbound_energy 0.0')
    replace_file_content(sub/'autodock_config.dpf', 'unbound_model extended', 'unbound_energy 0.0')

    run_autogrid4(sub/'autogrid_config.gpf', sub/'autogrid_output.glg')
    run_autodock4(sub/'autodock_config.dpf', sub/'autodock_output.dlg')


def single_dock(pdb_id, bond_res, work_dir, data_dir, total_runs=300):
    data_i_dir = Path(data_dir/pdb_id)
    work_i_dir = Path(work_dir/pdb_id)
    assert data_i_dir.exists(), f"Data directory for {pdb_id} does not exist: {data_i_dir}"
    os.makedirs(work_i_dir, exist_ok=True)

    with open(data_i_dir/'ligand_indices.txt', 'r') as f:
        ligand_indices = f.read().strip()
    # 预计算好的 grid box(中心=共价原子 SG;第一行 gridcenter,第二行 npts)
    grid_center, grid_npts = (data_i_dir/'grid_box.txt').read_text().splitlines()

    # preparation 阶段已校验并存好的若干随机构象;构象数直接由文件自动检测,无需再传一遍
    conf_sdfs = sorted(data_i_dir.glob('autodock_randomConform_*.sdf'))
    assert conf_sdfs, f'{pdb_id}: 找不到已验证构象 autodock_randomConform_*.sdf,请先重跑 preparation'
    ga_run = max(1, total_runs // len(conf_sdfs))   # 总次数按构象数均分(3 个 → 各 100,合计 ~300)

    # 每个构象在 work_i_dir/conf_k 下各产一份 autodock_output.dlg(grid 各算一次,~8s,可忽略)
    for k, conf_sdf in enumerate(conf_sdfs, 1):
        _dock_one_conformer(conf_sdf, work_i_dir / f'conf_{k}', bond_res, data_i_dir,
                            ligand_indices, grid_center, grid_npts, ga_run)


if __name__ == "__main__":

    data_dir = Path('/data3/lpw/dock_workdir/autodock_cov/preparation')
    work_dir = Path('/data3/lpw/dock_workdir/autodock_cov/result2')
    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')
    os.makedirs(work_dir, exist_ok=True)
    df_info = pd.read_csv(info_file)
    pdb_id = '4HCV'                                                               # '1A0L'
    total_runs = 300
    bond_info = df_info[df_info['pdb_id']==pdb_id]['bond'].values[0]              # OG SER D 195-C2 APA D 301'
    name , resName, chainID, resSeq = bond_info.split('-')[0].split(' ')
    bond_res = f'{chainID}:{resName}{resSeq}'                                     # D:SER195
    single_dock(pdb_id, bond_res, work_dir, data_dir,total_runs=total_runs)
