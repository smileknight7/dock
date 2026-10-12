import pandas as pd


'''
69个hard样本
['5DP7', '3C9W', '5HZE', '2HWP', '4HAX', '4GMX', '4YHF', '3W2P', '3OF9', '5W1Y', '4HCV', '4M1Y', '5TG2', '2EXB', '5V71', '5O8V', '2I03', '4M1S', '4M1T', '2HWO', '2QQ7', '2IPP', '2QLQ', '5DP5', '3ZVG', '4LUC', '2ZK2', '2WKE', '2YJ2', '4US2', '3LOK', '4QPS', '5TDI', '3RHY', '2ALV', '1NMS', '4LQM', '2YJB', '4PIQ', '5LCK', '2MLM', '2EX6', '2XYG', '4LYF', '5F2E', '4QQ5', '5GNK', '5J87', '4DMX', '2HXZ', '5V9O', '2ZU5', '2R6N', '2G6D', '2YJC', '4BS6', '5UG9', '5UGC', '4BQV', '5DAF', '1YK7', '4I24', '2XYP', '5ACB', '4PDZ', '3ZIM', '3V4J', '5LBG', '4YQV', '6FFN', '4M1W', '1NL6', '5J9Y', '3B1U', '2ZK4', '2OC0', '4WVF', '5HG9', '5HG5']

'''





def get_hard_pbd_ids(rmsd_file):
    df = pd.read_csv(rmsd_file)
    hard_pdb_ids = df[df['best_rmsd'] > 4]['pdb_id'].tolist()
    print('hard_pdb_ids:', hard_pdb_ids)
    print(len(hard_pdb_ids))



if __name__ == '__main__':
    dock_rmsd_file = '/data3/lpw/dock_workdir/autodock_cov/covdock_pose_analysis.csv'
    get_hard_pbd_ids(dock_rmsd_file)