
from pathlib import Path
import pandas as pd
from utils.cov_utils import parse_dlg_energies_multi



def analyze_pose_results(pose_dir, result_dir, info_file, pdb_ids=None):
    '''校验总体样本的rmsd结果，统计topk样本的成功率'''
    df_info = pd.read_csv(info_file)    
    if pdb_ids is not None:
        df_info = df_info[df_info['pdb_id'].isin(pdb_ids)]
    rows = []
    for _ , row in df_info.iterrows():
        pdb_id = row['pdb_id']
        output_i_dir = pose_dir/pdb_id
        rmsd_file = output_i_dir/f'{pdb_id}_docking_rmsd.csv'
        if not rmsd_file.exists():
            print(f"RMSD file {rmsd_file} does not exist. Please run get_rmsd() first.")
            continue
        df_rmsd = pd.read_csv(rmsd_file)
        # 把 DLG 结合能并成 energy 列,再按能量从低到高排序后回写(多构象:聚合 conf_*/autodock_output.dlg)
        energies = parse_dlg_energies_multi(result_dir/pdb_id)   # {pose_idx: energy},与 get_pose 重编号一致
        df_rmsd['energy'] = df_rmsd['pose_idx'].map(energies)
        df_rmsd = df_rmsd.sort_values('energy').reset_index(drop=True)
        df_rmsd.to_csv(rmsd_file, index=False)

        best_rmsd = df_rmsd['rmsd'].min()
        top_rmsd = df_rmsd.iloc[0]['rmsd']
        top_100_rmsd = df_rmsd.head(200)['rmsd'].min() if len(df_rmsd) >= 100 else None
        rows.append({'pdb_id': pdb_id, 'best_rmsd': best_rmsd, 'top_rmsd': top_rmsd, 'top_100_rmsd': top_100_rmsd })
        
    df_results = pd.DataFrame(rows)
    success_rate = (df_results['best_rmsd'] <= 2.0).mean()
    top_success_rate = (df_results['top_rmsd'] <= 2.0).mean() 
    top_100_success_rate = (df_results['top_100_rmsd'] <= 2.0).mean()
    print(f"Total samples: {len(df_results)}")
    print(f"oracle Success Rate: {success_rate:.2%}")
    print(f"top Success Rate: {top_success_rate:.2%}")
    print(f"top100 success rate: {top_100_success_rate:.2%}")
    
    df_results.to_csv(pose_dir.parent/'covdock_pose6_analysis.csv', index=False)

if __name__ == "__main__":
    pose_dir = Path('/data3/lpw/dock_workdir/autodock_cov/pose6')
    result_dir = Path('/data3/lpw/dock_workdir/autodock_cov/result6')   # DLG 所在目录
    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')
    complete_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/mopac_completed_200.txt')
    # pdb_ids = ['5DP7', '3C9W', '5HZE', '2HWP', '4HAX', '4GMX', '4YHF', '3W2P', '3OF9', '5W1Y', '4HCV', '4M1Y', '5TG2', '2EXB', '5V71', '5O8V', '2I03', '4M1S', '4M1T', '2HWO', '2QQ7', '2IPP', '2QLQ', '5DP5', '3ZVG', '4LUC', '2ZK2', '2WKE', '2YJ2', '4US2', '3LOK', '4QPS', '5TDI', '3RHY', '2ALV', '1NMS', '4LQM', '2YJB', '4PIQ', '5LCK', '2MLM', '2EX6', '2XYG', '4LYF', '5F2E', '4QQ5', '5GNK', '5J87', '4DMX', '2HXZ', '5V9O', '2ZU5', '2R6N', '2G6D', '2YJC', '4BS6', '5UG9', '5UGC', '4BQV', '5DAF', '1YK7', '4I24', '2XYP', '5ACB', '4PDZ', '3ZIM', '3V4J', '5LBG', '4YQV', '6FFN', '4M1W', '1NL6', '5J9Y', '3B1U', '2ZK4', '2OC0', '4WVF', '5HG9', '5HG5']
    # pdb_ids = None
    with open(complete_file, 'r') as f:
        pdb_ids = [line.strip() for line in f.readlines()]
    analyze_pose_results(pose_dir, result_dir, info_file=info_file, pdb_ids=pdb_ids)
