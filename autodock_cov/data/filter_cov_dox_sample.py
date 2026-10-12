import pandas as pd
from pathlib import Path
import os



'''
总结一下失败样本








338 个ID是重合的
想一想carsidock训练用的数据是什么样子的呢？有空要看下自蒸馏是怎么做的


GSA Success Rate  top300 : 85.80%
PM7 Success Rate top1: 57.40%
PM7 Success Rate (Top 10): 92.60%

 
# 260个成功对接的ID
GSA  Top300 Success Rate: 87.31%
PM7  Top1 Success Rate: 59.23%
PM7 Top 10 best Success Rate: 91.92%
Number of PDB IDs: 260



260
口袋划分按照pose来划分3A  50 pose
mopac Oracle    Success Rate: 46.72%
mopac Top-1     Success Rate: 19.31%
autodock Oracle Success Rate: 44.79%
autodock Top-1 Success Rate: 22.78%



口袋划分按照ref_ligand来划分
mopac Oracle    Success Rate: 42.69%
mopac Top-1     Success Rate: 18.46%
autodock Oracle Success Rate: 45.00%                # 修正grad口袋之后这个结果可以做到79.64%      增加采样初始化构象的话应该也可以显著提高orcale的比例
autodock Top-1 Success Rate: 23.08%                 # 37.05%


'''


def get_common_csv(covdocker_info_file, cov_dox_result_file):
    df_covdocker_info = pd.read_csv(covdocker_info_file)
    print(pd.ExcelFile(cov_dox_result_file).sheet_names)                        # 读写xlsx文件先要sheet名字
    df_cov_dox_result = pd.read_excel(cov_dox_result_file, sheet_name='Table4')     
    covdocker_pdb_ids = set(df_covdocker_info['pdb_id'].unique())
    cov_dox_pdb_ids = set(df_cov_dox_result['ID'].unique())

    common_pdb_ids = covdocker_pdb_ids.intersection(cov_dox_pdb_ids)
    df_common_covdocker = df_covdocker_info[df_covdocker_info['pdb_id'].isin(common_pdb_ids)]
    df_common_cov_dox = df_cov_dox_result[df_cov_dox_result['ID'].isin(values=common_pdb_ids)]
    df_common_covdocker.to_csv(output_dir / 'covdocker_info_common.csv', index=False)
    df_common_cov_dox.to_csv(output_dir / 'cov_dox_result_common.csv', index=False)
    print(f"Number of common PDB IDs: {len(common_pdb_ids)}")
    print(f"Common PDB IDs: {common_pdb_ids}")


def analysis_cov_dox_result(cov_dox_result_file, select_pdb_ids=None):
    '''input cov_dox csv output mopac top10 rmsd_success XO TOP1'''
    if select_pdb_ids is None:
        df_cov_dox_result = pd.read_csv(cov_dox_result_file)
        GSA_success_rate = (df_cov_dox_result['GSA Top300 BEST'] < 2.0).sum() / len(df_cov_dox_result)
        PM7_success_rate = (df_cov_dox_result['PM7 TOP1'] < 2.0).sum() / len(df_cov_dox_result)
        PM7_success_rate_top10 = (df_cov_dox_result['PM7 Top10 Best'] < 2.0).sum() / len(df_cov_dox_result)
        print(f"GSA  Top300 Success Rate: {GSA_success_rate:.2%}")
        print(f"PM7  Top1 Success Rate: {PM7_success_rate:.2%}")
        print(f"PM7 Top 10 best Success Rate: {PM7_success_rate_top10:.2%}")
    else:
        pdb_ids = set(select_pdb_ids)
        df_cov_dox_result = pd.read_csv(cov_dox_result_file)
        df_cov_dox_result = df_cov_dox_result[df_cov_dox_result['ID'].isin(pdb_ids)]
        GSA_success_rate = (df_cov_dox_result['GSA Top300 BEST'] < 2.0).sum() / len(df_cov_dox_result)
        PM7_success_rate = (df_cov_dox_result['PM7 TOP1'] < 2.0).sum() / len(df_cov_dox_result)
        PM7_success_rate_top10 = (df_cov_dox_result['PM7 Top10 Best'] < 2.0).sum() / len(df_cov_dox_result)
        print(f"GSA  Top300 Success Rate: {GSA_success_rate:.2%}")
        print(f"PM7  Top1 Success Rate: {PM7_success_rate:.2%}")
        print(f"PM7 Top 10 best Success Rate: {PM7_success_rate_top10:.2%}")
    
    print(f"Number of PDB IDs: {len(df_cov_dox_result)}")


def get_pdb_ids_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    select_ids = set(df['pdb_id'].unique())
    print('num_select_ids:', len(select_ids) )
    print(f"Number of PDB IDs in pose file: {list(select_ids)}")


def get_mopac_list(pose_file):
    '''挑选orcale中success的pose，但是score中没有排到top1的pose， 当前卡的阈值时rmsd4'''
    df_dock_pose = pd.read_csv(pose_file)
    mopac_list = df_dock_pose[(df_dock_pose['best_rmsd'] > 2.0) & (df_dock_pose['top_rmsd'] < 4.0)]['pdb_id'].tolist()
    print('mopac_list', mopac_list,
          '\n可以供mopac优化样本', len(mopac_list))
    return mopac_list


if __name__ == "__main__":
    covdocker_info_file = Path('/home/smileknight/data/covdocker/data/processed/dataset.filtered.random_split.csv')
    cov_dox_result_file = Path('/home/smileknight/dock_scripts/jm1c02007_si_002.xlsx')
    output_dir = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data')
    mopac_pose_dir =Path('/data3/lpw/dock_workdir/autodock_cov/mopac_pose2')
    pose_file  =Path('/data3/lpw/dock_workdir/autodock_cov/covdock_pose6_analysis.csv')
    
    
    ## 找到covdocker中和cov_dox中交叉的数据用于分析
    #get_common_csv(covdocker_info_file, cov_dox_result_file)


    ## 分析当前这些样本在PM7和XO上的Top表现
    # select_pdb_ids = []
    # for i in os.listdir(mopac_pose_dir):
    #     if Path(mopac_pose_dir / i).is_dir():
    #         select_pdb_ids.append(i)
    # import ipdb; ipdb.set_trace()
    # # select_pdb_ids = None
    # analysis_cov_dox_result(output_dir / 'cov_dox_result_common.csv',select_pdb_ids =select_pdb_ids )


    ## 获取pose_csv中id
    # get_pdb_ids_from_csv(pose_file)


    ## 获取mopac中成功但是没有排到top1的pose
    get_mopac_list(pose_file)