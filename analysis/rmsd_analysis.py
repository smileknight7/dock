
from meeko import PDBQTMolecule, RDKitMolCreate

from rdkit import Chem
import numpy as np
from einops import reduce
import os
import pandas as pd
from collections import defaultdict


# for pair data rmsd analysis is necessary



def get_rmsd(gt_mol, decoy_mol):
    matches = np.array(decoy_mol.GetSubstructMatches(gt_mol, uniquify=False, useChirality=False, maxMatches=100000))  # (m, n)
    gt_coords = gt_mol.GetConformer().GetPositions()                 # (n, 3)
    decoy_coords = decoy_mol.GetConformer().GetPositions()
    diff = decoy_coords[matches] - gt_coords                     # (m, n, 3)
    rmsds = np.sqrt((diff ** 2).sum(axis=-1).mean(axis=-1))      # (m,) 每种对应方式一个 RMSD
    return rmsds.min()

def read_dlg(dlg_path):
    pdbqt_mol = PDBQTMolecule.from_file(dlg_path, is_dlg=True, skip_typing=True)
    energies = [pose.score for pose in pdbqt_mol]
    top = pdbqt_mol._pose_data["cluster_leads_sorted"][0]                   # top1 的下标
    mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0]
    top_mol = Chem.Mol(mol, False, mol.GetConformers()[top].GetId()) 
    return top_mol, energies[top], top


def single_rmsd_culcalute(dlg_path, gt_path):
    mol, energy, top_id = read_dlg(dlg_path)
    gt_mol = Chem.SDMolSupplier(gt_path, removeHs=False)[0]
    min_rmsd = get_rmsd( gt_mol, mol)
    return energy, min_rmsd, top_id


def main(data_dir, result_dir):

    rows = []                                                       # 每个配体一行，最后一起转成表
    for pdb_id in sorted(os.listdir(result_dir)):
        gt_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
        dlg_path = os.path.join(result_dir, pdb_id, f"{pdb_id}.dlg")
        energy, rmsd, top_id = single_rmsd_culcalute(dlg_path, gt_path)
        rows.append({'pdb_id': pdb_id, 'rmsd': rmsd, 'energy': energy, 'top_id': top_id})

    df = pd.DataFrame(rows)
    success_rate = (df['rmsd'] < 2).mean()
    print(f"Success rate: {success_rate:.2%}")
    parent_dir = os.path.dirname(os.path.normpath(result_dir))
    df.to_csv(os.path.join(parent_dir, 'rmsd_analysis.csv'), index=False)

'''
或者可以写成这个形式

dic = defaultdict(list)
for pdb_id in sorted(os.listdir(result_dir)):
    ...
    energy, rmsd, top_id = single_rmsd_culcalute(dlg_path, gt_path)
    dic['pdb_id'].append(pdb_id)
    dic['rmsd'].append(rmsd)
    dic['energy'].append(energy)
    dic['top_id'].append(top_id)

'''



data_dir = '/data3/lpw/data/covdocker/data/processed/bonded'
result_dir = '/data3/lpw/dock_workdir/autodock_gpu/cov200_test/dock_result'
main(data_dir, result_dir)

print(read_dlg('/data3/lpw/dock_workdir/autodock_gpu/cov200_test/dock_result/7RBZ/7RBZ.dlg'))




# 当前状态下rmsd只有 12.8%