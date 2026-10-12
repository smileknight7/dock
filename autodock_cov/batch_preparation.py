from joblib import Parallel, delayed
from tqdm import tqdm   
from pathlib import Path
from tqdm_joblib import tqdm_joblib
import pandas as pd
from single_preparation import single_preparation
from traceback import format_exc

'''
需要隔离拆分io部分和算法部分的！！！！！

并行任务构建:
batch 部分读csv:        构建任务，写日志
run_one :               包装一层，隔离单样本失败
single_preparation :    算法层

['2Y59', '1SCW', '3HWN']     这三个样本处理失败


 1SCW   Chem.MolFromSmiles(product_smi) 返回 None → CSV 里 products 那条 SMILES RDKit 解析出错
 2Y59   找不到 aa 的第二连接原子                                      
 3HWN   找不到 ligand→aa 的连接键                                                              

还可以思考的问题，work_i_dir 是不是可以换个名字，当前使用单独一个函数来设计是否合适？

'''



def _run_one(pdb_id, bond, product_smi, data_dir, output_dir, n_conformers, max_attempts):
    """对单个样本做 preparation，捕获异常，返回 (pdb_id, ok, err_msg)。"""
    try:
        single_preparation(pdb_id, bond, product_smi, data_dir, output_dir, n_conformers=n_conformers, max_attempts=max_attempts)
        return pdb_id, True, ''
    except Exception:
        return pdb_id, False, format_exc()

def batch_preparation(info_file, data_dir, output_dir,pdb_ids=None, limit=None, n_jobs=4, n_conformers=3, max_attempts=20):  # 这里注意要改回来
    df_info = pd.read_csv(info_file)                                                                    # 始终先读;pdb_ids=None 时跑全量
    if pdb_ids is not None:
        df_info = df_info[~df_info['pdb_id'].isin(pdb_ids)]                                             # 这里直接bool取反就好了，这里注意要改回来
        # df_info = df_info[df_info['pdb_id'].isin(pdb_ids)]
    if limit is not None:
        df_info = df_info.head(limit)   
    # import ipdb; ipdb.set_trace()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for _, rows in df_info.iterrows():
        pdb_id = rows['pdb_id']
        bond = rows['bond']
        product_smi = rows['products']
        tasks.append((pdb_id, bond, product_smi, data_dir, output_dir))
   
    with tqdm_joblib(tqdm(desc='preparation', total=len(tasks))):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(pdb_id, bond, product_smi, data_dir, output_dir, n_conformers, max_attempts)
            for pdb_id, bond, product_smi, data_dir, output_dir in tasks
        )
    failed = [(pdb_id, err) for pdb_id, ok, err in results if not ok]
    print(f'\n完成: {len(results) - len(failed)}/{len(results)} 成功, {len(failed)} 失败')

    if failed:
            fail_log = output_dir / 'preparation_failed.log'
            with open(fail_log, 'w') as fp:
                for pdb_id, err in failed:
                        fp.write(f'===== {pdb_id} =====\n{err}\n')
            print('失败列表:', [pdb_id for pdb_id, _ in failed])
            print(f'详细错误已写入: {fail_log}')



if __name__ == '__main__':

    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')
    data_dir = Path('/data3/lpw/data/covdocker/data/processed/bonded')
    output_dir = Path('/data3/lpw/dock_workdir/autodock_cov/preparation5')
    
    limit = None           # 10
    n_conformers = 3       # 放到后续docking中尝试的次数
    max_attempts = 20      # 为达到n_conformers 尝试的采样次数
    pdb_ids = None  # ['5DP7', '3C9W', '5HZE', '2HWP', '4HAX', '4GMX', '4YHF', '3W2P', '3OF9', '5W1Y', '4HCV', '4M1Y', '5TG2', '2EXB', '5V71', '5O8V', '2I03', '4M1S', '4M1T', '2HWO', '2QQ7', '2IPP', '2QLQ', '5DP5', '3ZVG', '4LUC', '2ZK2', '2WKE', '2YJ2', '4US2', '3LOK', '4QPS', '5TDI', '3RHY', '2ALV', '1NMS', '4LQM', '2YJB', '4PIQ', '5LCK', '2MLM', '2EX6', '2XYG', '4LYF', '5F2E', '4QQ5', '5GNK', '5J87', '4DMX', '2HXZ', '5V9O', '2ZU5', '2R6N', '2G6D', '2YJC', '4BS6', '5UG9', '5UGC', '4BQV', '5DAF', '1YK7', '4I24', '2XYP', '5ACB', '4PDZ', '3ZIM', '3V4J', '5LBG', '4YQV', '6FFN', '4M1W', '1NL6', '5J9Y', '3B1U', '2ZK4', '2OC0', '4WVF', '5HG9', '5HG5']
    batch_preparation(info_file, data_dir, output_dir,pdb_ids=pdb_ids, limit=limit, n_jobs=8, n_conformers=n_conformers, max_attempts=max_attempts)