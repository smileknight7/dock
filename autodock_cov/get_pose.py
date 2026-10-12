
from pathlib import Path
from config import run_get_conformations, run_pdbq2pdb
from utils.cov_utils import _extract_ligand_pdb_block_from_prediction, _calc_template_order_rmsd
from rdkit import Chem
from rdkit.Chem import AllChem  
import pandas as pd
import os
from traceback import format_exc
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed 
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')   # 只关 warning
import shutil 


'''
这个构建data_i_dir是不是应该写再函数里面？？
data_i_dir 本事就是函数内部用，内部变量内部写应该是
外部变量外部用，内部变量函数内部构建




                
                
1. 修 RMSD 的原子对应（最高价值）——现在对 1/3 数据是瞎的。要让 pred 与参照有干净的 1:1 对应：要么改抽原子的方式，要么参照换成与对接配体一致的来源（注意 autodock_gt.sdf 比 pred 多 2 个 AA 锚原子、也不直接相等，需要专门对齐）。
2. 把结合能写进 rmsd.csv——现在要回头解析 .dlg 才能排 top1，直接在 get_rmsd 里给每个 pose 存一列 energy 更方便，也能直接出 top1/top5。
3. 可考虑用 AutoDock 的聚类（最大簇/簇代表）而非纯最低能量来定 top1。

现在能精确解释 B 类了。看 1CSB 的证据:

ligand_covalent.pdbqt       重原子 32   ← 完整共价配体
ligand_covalent_rigid.pdbqt 重原子 19   ← 大半被划成"刚性",冻进受体、不动、不进 dlg
ligand_covalent_flex.pdbqt  重原子  6   ← 只有这 6 个被当作可动配体去对接(2 个 torsion)





理论上flex部分应该就是ligand拆分后的部分
 prepare_flexreceptor4.py 这个脚本可能有问题

'''




B_truncation = ['1ATK','1AU3','1AU4','1CSB','1EWL','1EWM','1F29','1F2C','1F4C','1FH0','1NLJ','1YT7',
                '1ZLR','1ZPZ','2A5K','2FT2','2FYE','2PX6','2R9F','2TOD','2WAP','2XU4','2YJ9','3A3F',
                '3BPF','3HD3','3KW9','3LJ6','3S3Q','3W2Q','4AXM','4CDF','4D9U','4GS6','4HCU','4MZO',
                '4PE7','4PI3','4Q8I','4TWY','4XZ0','4YRT','5C91','5D11','5DP4','5DP9','5FED','5FX5',
                '5LCJ','5O8U','5P9J','5TH7','5V6V','6AY2']                           # 54 个: pose 只含柔性子集




def get_pose(pdb_id, dock_result_dir, output_dir):
    dock_result_i_dir = Path(dock_result_dir/pdb_id)
    assert dock_result_i_dir.exists(), f"Docking result directory {dock_result_i_dir} does not exist. Please run docking first."
    
    output_i_dir = Path(output_dir/pdb_id)
    os.makedirs(output_i_dir, exist_ok=True)                                                    # makedirs会递归创建路径上的目录，和mkdir不一样

    # 多构象:每个 conf_k 一份 dlg,各自抽 pose 后按 conf 顺序重编号成连续 1..N 落到 output_i_dir
    conf_dirs = sorted(dock_result_i_dir.glob('conf_*'), key=lambda p: int(p.name.split('_')[1]))
    assert conf_dirs, f"{dock_result_i_dir} 下没有 conf_* 子目录,请先重跑 docking"
    for old in output_i_dir.glob('res_ligand_pose_*.pdbqt'):                                    # 清掉上次残留,避免串号
        old.unlink()
    offset = 0
    for conf_dir in conf_dirs:
        tmp = output_i_dir / f'.tmp_{conf_dir.name}'                                            # 每个 conf 先抽到临时目录(各自从 1 编号),再按 offset 搬走
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        run_get_conformations(['-d', f'{conf_dir}/autodock_output.dlg', '-o', f'{tmp}/res_ligand_pose'], tmp)
        poses = sorted(tmp.glob('res_ligand_pose_*.pdbqt'), key=lambda p: int(p.stem.split('_')[-1]))
        for p in poses:
            j = int(p.stem.split('_')[-1])
            shutil.copy(p, output_i_dir / f'res_ligand_pose_{offset + j}.pdbqt')
        offset += len(poses)
        shutil.rmtree(tmp)

def get_rmsd(pdb_id, data_dir, output_dir, dock_nums):
    '''将dlg中提取的pose转成pdb再计算rmsd'''
    data_i_dir = Path(data_dir/pdb_id)
    output_i_dir = Path(output_dir/pdb_id)
    assert output_i_dir.exists(), f"Output directory {output_i_dir} does not exist. Please run get_pose() first."
    
    ligand_gt_file = f'{data_i_dir}/{pdb_id}_ligand.sdf'
    pocket_file = f'{data_i_dir}/{pdb_id}_10Apocket.pdb'
    shutil.copy(ligand_gt_file, output_i_dir/f'{pdb_id}_ligand.sdf')  
    shutil.copy(pocket_file, output_i_dir/f'{pdb_id}_10Apocket.pdb')
    rows = []
    # 直接按实际抽到的 pose 文件遍历(多构象合计,数量由文件决定,不再依赖 dock_nums)
    pose_files = sorted(output_i_dir.glob('res_ligand_pose_*.pdbqt'), key=lambda p: int(p.stem.split('_')[-1]))
    for p in pose_files:
        i = int(p.stem.split('_')[-1])
        pred_ligand_file = f'{output_i_dir}/res_ligand_pose_{i}.pdb'
        run_pdbq2pdb(['-f', f'res_ligand_pose_{i}.pdbqt'], output_i_dir)
        pred_pdb_block=_extract_ligand_pdb_block_from_prediction(pred_ligand_file)
        pred_mol = Chem.MolFromPDBBlock(pred_pdb_block, removeHs=True)
        Chem.MolToPDBFile(pred_mol, str(output_i_dir/f'pred_ligand_pose_{i}.pdb'))                # c++ 不支持path数据类型（但是rdkit有些数据好像是支持的，不清楚这个东西咋回事）
        ligand_gt_mol = Chem.SDMolSupplier(str(ligand_gt_file))[0]
        assert pred_mol.GetNumAtoms() == ligand_gt_mol.GetNumAtoms(), f"Predicted ligand has {pred_mol.GetNumAtoms()} atoms, but GT ligand has {ligand_gt_mol.GetNumAtoms()} atoms"
        pred_mol = AllChem.AssignBondOrdersFromTemplate(ligand_gt_mol, pred_mol)
        matches = pred_mol.GetSubstructMatches(ligand_gt_mol, uniquify=False)
        rmsd_value, best_match = _calc_template_order_rmsd(ligand_gt_mol, pred_mol, matches)
        rows.append({'pdb_id': pdb_id, 'pose_idx': i, 'rmsd': rmsd_value})
    df = pd.DataFrame(rows)
    df.to_csv(output_i_dir/f'{pdb_id}_docking_rmsd.csv', index=False)
    # print(df)

def _run_one(pdb_id, dock_result_dir, output_dir, data_dir, dock_nums):
    RDLogger.DisableLog('rdApp.warning')   # loky worker 是独立进程，模块级第14行不会在子进程执行，必须在 worker 内部再关一次
    try:
        get_pose(pdb_id,dock_result_dir, output_dir)
        get_rmsd(pdb_id, data_dir, output_dir, dock_nums)
        
        return pdb_id, True, ''
    except Exception as e:
        return pdb_id, False, format_exc()


def batch_get_pose(info_file, dock_result_dir, output_dir, data_dir, pdb_ids,  dock_nums, limit=10, n_jobs=4):
    if pdb_ids is not None:
        df_info = pd.read_csv(info_file)
        df_info = df_info[df_info['pdb_id'].isin(pdb_ids)]
        # df_info = df_info[~df_info['pdb_id'].isin(pdb_ids)]
    if limit is not None:
        df_info = df_info.head(limit)   

    tasks = []
    for _, rows in df_info.iterrows():
        pdb_id = rows['pdb_id']
        tasks.append((pdb_id, dock_result_dir, output_dir, data_dir, dock_nums))
   
    with tqdm_joblib(tqdm(desc='get_pose', total=len(tasks))):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(pdb_id, dock_result_dir, output_dir, data_dir, dock_nums)
            for pdb_id, dock_result_dir, output_dir, data_dir, dock_nums in tasks
        )
    failed = [(pdb_id, err) for pdb_id, ok, err in results if not ok]
    print(f'\n完成: {len(results) - len(failed)}/{len(results)} 成功, {len(failed)} 失败')

    if failed:
            fail_log = output_dir / 'get_pose_failed.log'
            with open(fail_log, 'w') as fp:
                for pdb_id, err in failed:
                        fp.write(f'===== {pdb_id} =====\n{err}\n')
            print('失败列表:', [pdb_id for pdb_id, _ in failed])
            print(f'详细错误已写入: {fail_log}')





if __name__ == "__main__":

    data_dir = Path('/data3/lpw/data/covdocker/data/processed/bonded')
    dock_result_dir = Path('/data3/lpw/dock_workdir/autodock_cov/result6')
    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')
    output_dir = Path('/data3/lpw/dock_workdir/autodock_cov/pose6')
    dock_nums =  300
    limit = None
    df_info = pd.read_csv(info_file)
    single_dock = False
    # pdb_ids = ['5DP7', '3C9W', '5HZE', '2HWP', '4HAX', '4GMX', '4YHF', '3W2P', '3OF9', '5W1Y', '4HCV', '4M1Y', '5TG2', '2EXB', '5V71', '5O8V', '2I03', '4M1S', '4M1T', '2HWO', '2QQ7', '2IPP', '2QLQ', '5DP5', '3ZVG', '4LUC', '2ZK2', '2WKE', '2YJ2', '4US2', '3LOK', '4QPS', '5TDI', '3RHY', '2ALV', '1NMS', '4LQM', '2YJB', '4PIQ', '5LCK', '2MLM', '2EX6', '2XYG', '4LYF', '5F2E', '4QQ5', '5GNK', '5J87', '4DMX', '2HXZ', '5V9O', '2ZU5', '2R6N', '2G6D', '2YJC', '4BS6', '5UG9', '5UGC', '4BQV', '5DAF', '1YK7', '4I24', '2XYP', '5ACB', '4PDZ', '3ZIM', '3V4J', '5LBG', '4YQV', '6FFN', '4M1W', '1NL6', '5J9Y', '3B1U', '2ZK4', '2OC0', '4WVF', '5HG9', '5HG5']
    
    pdb_ids =['5HG5', '2OP3', '2XU1', '2YJC', '3ZIM', '3SVV', '2FQ9', '4M1Y', '3KWB', '3BWK', '1NPZ', '3A3E', '5E7R', '3IKA', '5HG7', '2HWP', '5J9Z', '2H5S', '4LQM', '2EX8', '2XU5', '4LYH', '1VM1', '5LCJ', '5P9M', '3KW9', '2FT2', '5DP4', '2QLQ', '5DAF', '1ZSL', '2HXZ', '4YRS', '3CG5', '5DP7', '1F2A', '4BQV', '4HCU', '5GNK', '5VQX', '1FH0', '3KJN', '3V4J', '3OF9', '2YJB', '1RWN', '5WFJ', '5TTS', '1BMQ', '5J8I', '4M1S', '1TU6', '5F2E', '4HAV', '2R6N', '1F2B', '5C1Y', '2J8Y', '4XCU', '1AIM', '2YJ8', '5UG9', '5FX5', '3UPP', '5O8U', '3A3F', '3H0E', '5VQV', '5TH7', '2FRA', '5O8V', '2GH5', '4THI', '4BSQ', '3B1U', '4PIS', '4L1O', '1AYW', '3ZVG', '3LOK', '4M21', '4LV6', '5D11', '3BFC', '4BS5', '3BM8', '1NL6', '4PE4', '1EWM', '5VQY', '4G5J', '2OC0', '1AU4', '2YJ9', '2IPP', '5FED', '4PDZ', '3BFG', '5HG9', '5TTU', '5MAJ', '4YQU', '2HWO', '4UUQ', '3LJ6', '4DCD', '3SN8', '1CEF', '3A3I', '5V9O', '2ZU5', '1ZPZ', '2OP9', '4DKI', '5IYT', '1THE', '1NFZ', '8EST', '1EWO', '3N3G', '4XZ0', '2EXB', '2ALV', '3S3Q', '3MZD', '2EX9', '1RHJ', '4YQM', '4CDC', '5LBG', '5JH6', '5J9Y', '1GMY', '4HCV', '1OCE', '2I03', '1U9W', '4WVF', '3V6R', '3HHA', '5TDI', '4D9U', '4YHF', '4ZZM', '1NQC', '2R4B', '5W1Y', '4TWY', '2ZK4', '5V71', '5MJB', '2PX6', '5C91', '3C9W', '5J7S', '4MZS', '3BEB', '1AU3', '1AYU', '5TOZ', '1U9V', '5JK3', '4LYJ', '5KRE', '5FX6', '4CDE', '5DP9', '5EA9', '5HG8', '2YJ2', '4QQC', '5DP8', '2WKE', '5LCK', '5EPH', '2ZK5', '3UPN', '3BJM', '2XYP', '3UPO', '5TTV', '2IJN', '4I24', '3O6T', '3O1G', '5E93', '5VND', '1RHM', '2JBF', '4QQ5', '4ZZO', '1HVY', '3BLT', '3OVX', '5UGC', '3W2P', '5MQY', '2F7D', '4HAY', '2QQ7', '4BS6', '3BLU', '3ZFZ', '1JU6', '6FFN', '2WAP', '4Z16', '4WX7', '1CEG', '2DW5', '4ONM', '5TG2', '3RHY', '1H8I', '4M22', '4WX4', '5DAD', '5N19', '2BDL', '4PIQ', '4X0T', '2V35', '5C5O', '3ZV9', '5HZE', '2XYG', '5C1X', '4YQV', '1F2C', '4US1', '5L6P', '1RE1', '4CDD', '1F29', '4X6J', '5GIT', '4PE0', '1CVZ', '1RWP', '3HD3', '3N4C', '2GX4', '2FRQ', '2EX6', '4DMX', '1F4C', '2AX1', '5P9K', '5ACB', '4CDF', '4M1W', '3W2Q', '4M1T', '4DMY', '3PAE', '5V6V', '5DPA', '3W2T', '4HAX', '2G8E', '2TOD', '1YK7', '4YRT', '2A5K', '5F02', '4YM9', '4M1O', '4LUC', '4QPS', '4LYF', '1EWP', '3KJF', '5J87', '1AU0', '2AUX', '2VGJ', '3KWZ', '4Q8I']
    if single_dock: # 方便调试单个样本
        pdb_id = '4HCV'   # '4HAX
        get_pose(pdb_id, dock_result_dir, output_dir)
        get_rmsd(pdb_id, data_dir, output_dir, dock_nums)
    else:
        batch_get_pose(info_file, dock_result_dir, output_dir, data_dir, pdb_ids=pdb_ids, dock_nums=dock_nums, limit=limit, n_jobs=4)