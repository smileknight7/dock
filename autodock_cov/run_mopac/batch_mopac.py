from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from pathlib import Path
from collections import defaultdict
import pandas as pd
from traceback import format_exc

from run_mopac.single_mopac import (
    list_pose_indices, prepare_pose_dirs, run_one_pose, DEFAULT_MOPAC_KW,
)
'''
40核处理26h！！！！

失败样本
['3B0R', '4GMX', '2TOD', '2FYE', '1ZPZ', '4XZ0', '2R9F', '1QHR', '3KJN', '4PI3', '1FH0', '4MZO', '2Y59', '2XU4', '1NLJ', '1EWM', '5DP9', '3W2Q', '3KW9', '5P9J', '3BLT', '1F29', '6AY2', '1NPZ', '5O8U', '1F4C', '1ZLR', '1EWL', '4Q8I', '5TH7', '3RHY', '1QJ7', '2FT2', '1AU4', '1SCW', '4YRT', '5V6V', '2A5K', '4D9U', '5LCJ', '1ATK', '1AU3', '2FUD', '1F2C', '3LJ6', '3S3Q', '4PE4', '4PE7', '1QJ1', '1QJ6', '2G6D', '4HCU', '4WX6', '4ONM', '3HWN', '4GS6', '5D11', '2PX6', '5DP4', '3A3F', '1YT7', '3HD3', '1W8Y', '2YJ9', '4AXM', '2XLN', '4PDZ', '3V4J', '5C91', '4CDF', '2XK1', '2WAP', '4TWY', '5FED', '5FX5', '1CSB', '3BPF', '4THI', '2F9A']
'''

def _run_one_pose(pdb_id, idx, mopac_dir, result_dir, mopac_pose_dir, mopac_traj_dir,
                  mopac_kw, run):
    """pose 级 worker：跑单个 pose，捕获异常，返回 (pdb_id, idx, ok, err_msg)。"""
    try:
        run_one_pose(pdb_id, idx, mopac_dir, result_dir, mopac_pose_dir, mopac_traj_dir,
                     mopac_kw=mopac_kw, run=run)
        return pdb_id, idx, True, ''
    except Exception:
        return pdb_id, idx, False, format_exc()


def batch_mopac(info_file, data_dir, mopac_dir, result_dir, mopac_pose_dir, mopac_traj_dir, pdb_ids=None,
                sample_nums=None, pose_nums=10, mopac_kw=DEFAULT_MOPAC_KW, run=True,
                limit=None, n_jobs=4):
    """在 pose 层并行跑 MOPAC + 提取 pose（每个 (pdb, pose) 是一个独立任务）。

    并行度由 n_jobs 控制；MOPAC 自身 THREADS=1，故每个 job 占 1 核。
    相比 pdb 层并行，pose 层展开让单个 pdb 的多个 pose 也能并行，
    即使只有少量 pdb 也能吃满多核。
    """
    df_info = pd.read_csv(info_file)
    if limit is not None:
        df_info = df_info.head(limit)
    if pdb_ids is not None: 
        df_info = df_info[df_info['pdb_id'].isin(pdb_ids)]
    
    
    Path(mopac_pose_dir).mkdir(parents=True, exist_ok=True)
    # 1) 每个 pdb 一次性准备（建目录 + 拷贝），并把所有 (pdb, idx) 展开成 pose 任务
    pose_tasks = []
    prep_failed = []
    for _, row in df_info.iterrows():
        pdb_id = row['pdb_id']
        bond = row['bond']                                   # 'OG SER D 195-C2 APA D 301'
        name, res_name, chain_id, res_seq = bond.split('-')[0].split(' ')
        bond_res = f'{chain_id}:{res_name}-{res_seq}-{name}'  # 'D:SER-195-OG'
        try:
            prepare_pose_dirs(pdb_id, bond_res, data_dir, result_dir, mopac_pose_dir, mopac_traj_dir)
            idxs = list_pose_indices(pdb_id, mopac_dir, pose_nums, sample_nums)
        except Exception:
            prep_failed.append((pdb_id, format_exc()))
            continue
        pose_tasks.extend((pdb_id, idx) for idx in idxs)

    n_pdb = df_info.shape[0] - len(prep_failed)
    print(f'待处理: {n_pdb} 个 pdb, 共 {len(pose_tasks)} 个 pose 任务')

    # 2) pose 级并行
    with tqdm_joblib(tqdm(desc='mopac', total=len(pose_tasks))):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one_pose)(
                pdb_id, idx, mopac_dir, result_dir, mopac_pose_dir, mopac_traj_dir,
                mopac_kw, run,
            )
            for (pdb_id, idx) in pose_tasks
        )

    # 3) 聚合：按 pdb 归并失败的 pose
    pose_failed = [(pdb_id, idx, err) for pdb_id, idx, ok, err in results if not ok]
    fail_by_pdb = defaultdict(list)
    for pdb_id, idx, err in pose_failed:
        fail_by_pdb[pdb_id].append(idx)

    n_pose_ok = len(results) - len(pose_failed)
    print(f'\nPose 完成: {n_pose_ok}/{len(results)} 成功, {len(pose_failed)} 失败'
          f'（涉及 {len(fail_by_pdb)} 个 pdb）')
    if prep_failed:
        print(f'另有 {len(prep_failed)} 个 pdb 准备阶段失败: {[p for p, _ in prep_failed]}')

    if pose_failed or prep_failed:
        fail_log = Path(mopac_pose_dir) / 'mopac_failed.log'
        with open(fail_log, 'w') as fp:
            for pdb_id, err in prep_failed:
                fp.write(f'===== {pdb_id} [prepare] =====\n{err}\n')
            for pdb_id, idx, err in pose_failed:
                fp.write(f'===== {pdb_id} pose {idx} =====\n{err}\n')
        if fail_by_pdb:
            print('失败 pose（按 pdb）:', {p: sorted(ids) for p, ids in fail_by_pdb.items()})
        print(f'详细错误已写入: {fail_log}')


if __name__ == '__main__':
    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')
    data_dir = Path('/data3/lpw/data/covdocker/data/processed/bonded')
    mopac_dir = Path('/data3/lpw/dock_workdir/autodock_cov/mopac3')
    result_dir = Path('/data3/lpw/dock_workdir/autodock_cov/mopac_result3')
    mopac_pose_dir = Path('/data3/lpw/dock_workdir/autodock_cov/mopac_pose3')
    mopac_traj_dir = Path('/data3/lpw/dock_workdir/autodock_cov/mopac_traj3')
    limit = None
    n_jobs = 30
    pdb_ids = ['3C9W', '3CG5', '5TG2', '5E93', '5GIT', '2WKE', '3KWZ', '3LOK', '4YRT', '4HAY', '3N3G', '5DAF', '3ZFZ', '3UPO', '1U9V', '2OP9', '4THI']
    batch_mopac(info_file, data_dir, mopac_dir, result_dir, mopac_pose_dir, mopac_traj_dir,pdb_ids=pdb_ids,
                sample_nums=None, pose_nums=50, run=True, limit=limit, n_jobs=n_jobs)
