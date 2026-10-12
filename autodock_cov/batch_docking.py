from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from pathlib import Path
import pandas as pd
from traceback import format_exc

from single_dock import single_dock

'''

failed_docking     = ['1QHR', '2Y59', '1QJ7', '1SCW', '2FUD',
                      '1QJ1', '1QJ6', '3HWN', '1W8Y', '2XLN', '2XK1', '2F9A']              # 12 个

prepareCovalent.py 不认识该残基（residues[resName]['type'] KeyError）——adcovalent 工具对非标准/特殊残基的限制      │ 1QHR, 1QJ7, 1QJ1, 1QJ6, 2F9A │
缺 ligand_indices.txt —— 这 3 个正是上面 preparation 失败的样本，没产物所以 docking 第 21 行 open 失败，连锁失败    │ 2Y59, 1SCW, 3HWN             │
autogrid4 不支持原子类型 B（硼）——配体含硼，AD4 参数库默认没有                                                     │ 2XLN, 2XK1                   │
autogrid4 不支持原子类型 Si（硅）                                                                                │ 2FUD                         │
prepare_receptor4 原子杂化识别失败（babel_type not found）                                                       │ 1W8Y                         │                      



'''

def _run_one(pdb_id, bond_res, work_dir, data_dir, total_runs):
    """对单个样本做 docking，捕获异常，返回 (pdb_id, ok, err_msg)。"""
    try:
        single_dock(pdb_id, bond_res, work_dir, data_dir, total_runs=total_runs)
        return pdb_id, True, ''
    except Exception:
        return pdb_id, False, format_exc()


def batch_docking(info_file, data_dir, work_dir, limit=None, n_jobs=4, pdb_ids=None, total_runs=300):
    df_info = pd.read_csv(info_file)
    if pdb_ids is not None:
        # df_info = df_info[~df_info['pdb_id'].isin(pdb_ids)]                         # 这里取反，注意修改回来
        df_info = df_info[df_info['pdb_id'].isin(pdb_ids)]
    if limit is not None:
        df_info = df_info.head(limit)
    work_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in df_info.iterrows():
        pdb_id = row['pdb_id']
        bond = row['bond']
        # 从 bond 解析出 chain:resNAMEresNUM，例如 D:SER195
        name, resName, chainID, resSeq = bond.split('-')[0].split(' ')
        bond_res = f'{chainID}:{resName}{resSeq}'
        tasks.append((pdb_id, bond_res, work_dir, data_dir))

    with tqdm_joblib(tqdm(desc='docking', total=len(tasks))):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(pdb_id, bond_res, work_dir, data_dir, total_runs)
            for pdb_id, bond_res, work_dir, data_dir in tasks
        )

    failed = [(pdb_id, err) for pdb_id, ok, err in results if not ok]
    print(f'\n完成: {len(results) - len(failed)}/{len(results)} 成功, {len(failed)} 失败')

    if failed:
        fail_log = work_dir / 'docking_failed.log'
        with open(fail_log, 'w') as fp:
            for pdb_id, err in failed:
                fp.write(f'===== {pdb_id} =====\n{err}\n')
        print('失败列表:', [pdb_id for pdb_id, _ in failed])
        print(f'详细错误已写入: {fail_log}')


if __name__ == '__main__':
    data_dir = Path('/data3/lpw/dock_workdir/autodock_cov/preparation5')   # preparation 的产物在这里
    work_dir = Path('/data3/lpw/dock_workdir/autodock_cov/result6')        # docking 结果写这里
    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')
    limit = None  # 如果想测试前 N 个样本，可以设置为一个整数
    n_jobs = 40
    total_runs = 300   # 每个样本的总 GA 次数;single_dock 会按 preparation 生成的构象数自动均分(3 构象 → 各 100)
    pdb_ids =['5HG5', '2OP3', '2XU1', '2YJC', '3ZIM', '3SVV', '2FQ9', '4M1Y', '3KWB', '3BWK', '1NPZ', '3A3E', '5E7R', '3IKA', '5HG7', '2HWP', '5J9Z', '2H5S', '4LQM', '2EX8', '2XU5', '4LYH', '1VM1', '5LCJ', '5P9M', '3KW9', '2FT2', '5DP4', '2QLQ', '5DAF', '1ZSL', '2HXZ', '4YRS', '3CG5', '5DP7', '1F2A', '4BQV', '4HCU', '5GNK', '5VQX', '1FH0', '3KJN', '3V4J', '3OF9', '2YJB', '1RWN', '5WFJ', '5TTS', '1BMQ', '5J8I', '4M1S', '1TU6', '5F2E', '4HAV', '2R6N', '1F2B', '5C1Y', '2J8Y', '4XCU', '1AIM', '2YJ8', '5UG9', '5FX5', '3UPP', '5O8U', '3A3F', '3H0E', '5VQV', '5TH7', '2FRA', '5O8V', '2GH5', '4THI', '4BSQ', '3B1U', '4PIS', '4L1O', '1AYW', '3ZVG', '3LOK', '4M21', '4LV6', '5D11', '3BFC', '4BS5', '3BM8', '1NL6', '4PE4', '1EWM', '5VQY', '4G5J', '2OC0', '1AU4', '2YJ9', '2IPP', '5FED', '4PDZ', '3BFG', '5HG9', '5TTU', '5MAJ', '4YQU', '2HWO', '4UUQ', '3LJ6', '4DCD', '3SN8', '1CEF', '3A3I', '5V9O', '2ZU5', '1ZPZ', '2OP9', '4DKI', '5IYT', '1THE', '1NFZ', '8EST', '1EWO', '3N3G', '4XZ0', '2EXB', '2ALV', '3S3Q', '3MZD', '2EX9', '1RHJ', '4YQM', '4CDC', '5LBG', '5JH6', '5J9Y', '1GMY', '4HCV', '1OCE', '2I03', '1U9W', '4WVF', '3V6R', '3HHA', '5TDI', '4D9U', '4YHF', '4ZZM', '1NQC', '2R4B', '5W1Y', '4TWY', '2ZK4', '5V71', '5MJB', '2PX6', '5C91', '3C9W', '5J7S', '4MZS', '3BEB', '1AU3', '1AYU', '5TOZ', '1U9V', '5JK3', '4LYJ', '5KRE', '5FX6', '4CDE', '5DP9', '5EA9', '5HG8', '2YJ2', '4QQC', '5DP8', '2WKE', '5LCK', '5EPH', '2ZK5', '3UPN', '3BJM', '2XYP', '3UPO', '5TTV', '2IJN', '4I24', '3O6T', '3O1G', '5E93', '5VND', '1RHM', '2JBF', '4QQ5', '4ZZO', '1HVY', '3BLT', '3OVX', '5UGC', '3W2P', '5MQY', '2F7D', '4HAY', '2QQ7', '4BS6', '3BLU', '3ZFZ', '1JU6', '6FFN', '2WAP', '4Z16', '4WX7', '1CEG', '2DW5', '4ONM', '5TG2', '3RHY', '1H8I', '4M22', '4WX4', '5DAD', '5N19', '2BDL', '4PIQ', '4X0T', '2V35', '5C5O', '3ZV9', '5HZE', '2XYG', '5C1X', '4YQV', '1F2C', '4US1', '5L6P', '1RE1', '4CDD', '1F29', '4X6J', '5GIT', '4PE0', '1CVZ', '1RWP', '3HD3', '3N4C', '2GX4', '2FRQ', '2EX6', '4DMX', '1F4C', '2AX1', '5P9K', '5ACB', '4CDF', '4M1W', '3W2Q', '4M1T', '4DMY', '3PAE', '5V6V', '5DPA', '3W2T', '4HAX', '2G8E', '2TOD', '1YK7', '4YRT', '2A5K', '5F02', '4YM9', '4M1O', '4LUC', '4QPS', '4LYF', '1EWP', '3KJF', '5J87', '1AU0', '2AUX', '2VGJ', '3KWZ', '4Q8I']

    batch_docking(info_file, data_dir, work_dir, limit=limit, n_jobs=n_jobs, pdb_ids=pdb_ids, total_runs=total_runs)
