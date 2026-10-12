from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from pathlib import Path
import pandas as pd
from traceback import format_exc

from run_mopac.single_preparation_mopac import single_preparation


def _inputs_ready(pdb_id, pose_dir, data_dir, result_dir):
    """检查某个 pdb 的 mopac preparation 输入是否齐全（pose / rigid pdbqt / ligand sdf）。"""
    pose_i_dir = pose_dir / pdb_id
    pose_ok = pose_i_dir.is_dir() and any(pose_i_dir.glob('res_ligand_pose_*.pdb'))
    rigid_ok = (result_dir / pdb_id / 'conf_1'/'pocket_rigid.pdbqt').exists()                            # 这里也修改过一下（因为现在采样多个随机构象了）
    sdf_ok = (data_dir / pdb_id / f'{pdb_id}_ligand.sdf').exists()
    return pose_ok and rigid_ok and sdf_ok


def _run_one(pdb_id, bond_res, pose_dir, data_dir, mopac_dir, result_dir, sample_nums):
    """对单个 pdb_id 做 mopac preparation，捕获异常，返回 (pdb_id, ok, err_msg)。"""
    from rdkit import RDLogger                          # 这样保证批量静音但是单点不静音(这样写的话每个进程都会加载一遍)
    RDLogger.DisableLog('rdApp.warning')
    try:
        single_preparation(pdb_id, bond_res, pose_dir, data_dir, mopac_dir, result_dir, sample_nums)
        return pdb_id, True, ''
    except Exception:
        return pdb_id, False, format_exc()


def batch_preparation_mopac(info_file, pose_dir, data_dir, mopac_dir, result_dir,pdb_ids=None,
                            sample_nums=10, limit=None, n_jobs=4, skip_missing=True):
    df_info = pd.read_csv(info_file)
    
    if pdb_ids is not None:
        df_info = df_info[df_info['pdb_id'].isin(pdb_ids)]
    if limit is not None:
        df_info = df_info.head(limit)


    mopac_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    skipped = []
    for _, row in df_info.iterrows():
        pdb_id = row['pdb_id']
        # 输入未就绪的直接跳过（很多 pose/result 还没生成），避免无谓失败
        if skip_missing and not _inputs_ready(pdb_id, pose_dir, data_dir, result_dir):
            skipped.append(pdb_id)
            continue
        bond = row['bond']                                   # 'OG SER D 195-C2 APA D 301'
        name, res_name, chain_id, res_seq = bond.split('-')[0].split(' ')
        bond_res = f'{chain_id}:{res_name}-{res_seq}-{name}'  # 'D:SER-195-OG'
        tasks.append((pdb_id, bond_res, pose_dir, data_dir, mopac_dir, result_dir, sample_nums))

    if skipped:
        print(f'跳过 {len(skipped)} 个输入未就绪的 pdb（skip_missing=True）')
    print(f'待处理: {len(tasks)} 个 pdb')
    with tqdm_joblib(tqdm(desc='mopac-prep', total=len(tasks))):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(*task) for task in tasks
        )

    failed = [(pdb_id, err) for pdb_id, ok, err in results if not ok]
    print(f'\n完成: {len(results) - len(failed)}/{len(results)} 成功, {len(failed)} 失败')

    if failed:
        fail_log = mopac_dir / 'mopac_preparation_failed.log'
        with open(fail_log, 'w') as fp:
            for pdb_id, err in failed:
                fp.write(f'===== {pdb_id} =====\n{err}\n')
        print('失败列表:', [pdb_id for pdb_id, _ in failed])
        print(f'详细错误已写入: {fail_log}')


if __name__ == '__main__':
    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')
    pose_dir = Path('/data3/lpw/dock_workdir/autodock_cov/pose6')
    data_dir = Path('/data3/lpw/data/covdocker/data/processed/bonded')
    mopac_dir = Path('/data3/lpw/dock_workdir/autodock_cov/mopac5')
    result_dir = Path('/data3/lpw/dock_workdir/autodock_cov/result6')
    limit = 100
    n_jobs = 25
    sample_nums = 100
    pdb_ids = None # ['3C9W', '3CG5', '5TG2', '5E93', '5GIT', '2WKE', '3KWZ', '3LOK', '4YRT', '4HAY', '3N3G', '5DAF', '3ZFZ', '3UPO', '1U9V', '2OP9', '4THI']

    batch_preparation_mopac(
        info_file, pose_dir, data_dir, mopac_dir, result_dir, pdb_ids=pdb_ids,
        sample_nums=sample_nums, limit=limit, n_jobs=n_jobs, skip_missing=True,
    )
    if limit is not None:
        pdb_ids = pd.read_csv(info_file).head(limit)['pdb_id'].tolist()
        print(f'已处理的 pdb_id: {pdb_ids}')
        # ['3BEB', '5DP7', '5IYT', '3C9W', '2OP3', '2G8E', '5HZE', '2EX8', '2HWP', '4LYH'] 测试使用