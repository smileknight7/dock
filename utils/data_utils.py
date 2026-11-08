
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb
import  pandas as pd
import numpy as np
from scipy import spatial
from meeko import MoleculePreparation, PDBQTWriterLegacy
from pathlib import  Path
import math
from utils.config import GRID_SPACING







def canonicalize_mol(mol):
    Chem.MolToSmiles(mol)
    new_to_old = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    canonical_mol = Chem.RenumberAtoms(mol, new_to_old)          # 重排,保留 conformer/属性
    return canonical_mol

    
def generate_conformer(mol) -> Chem.Mol:
    mol = canonicalize_mol(mol)
    mol.RemoveAllConformers()  
    mol = Chem.AddHs(mol) 
    # if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:              #  EmbedMolecule 默认 clearConfs= True   
    #     raise ValueError("3D embedding failed")
    params = AllChem.ETKDGv3()
    params.randomSeed = 42                 
    if AllChem.EmbedMolecule(mol, params) < 0:   
        raise ValueError("3D embedding failed")
    # 构象优化
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    elif AllChem.UFFHasAllMoleculeParams(mol):
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mol

# 写pdbqt
def prepare_ligand(mol, output_file):
    mk_prep = MoleculePreparation()
    mol_status = mk_prep.prepare(mol)[0]
    text, ok, err = PDBQTWriterLegacy.write_string(mol_status)
    if not ok:
        raise RuntimeError(f"Meeko PDBQT write failed: {err}")
    Path(output_file).write_text(text)



def parse_protein(pdb_path):
    '''returns a list of dicts with keys: chain, res_id, res_name, x, y, z, element, line'''
    ppdb = PandasPdb().read_pdb(str(pdb_path))
    df_protein = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]], ignore_index=True)         # dim=0 拼接成一列

    return df_protein


# 解决pocket读取过程中默认H出错
def remove_protein_H(protein_file):
    mol = Chem.MolFromPDBFile(
    str(protein_file), sanitize=False, removeHs=False
)
    if mol is None:
        raise ValueError(f"Cannot read receptor: {protein_file}")

    mol = Chem.RemoveAllHs(mol, sanitize=False)
    pdb_str = Chem.MolToPDBBlock(mol)
    return pdb_str



# 划分口袋
def select_pocket(df_protein, center_coor, dis, any_atom=False):   # any_atom=True,  center_coor---->ligand_pos
    df_pocket = df_protein.copy()
    if not any_atom:
        df_tmp = df_pocket[df_pocket['atom_name'] == 'CA'].copy()
    else:
        df_tmp = df_pocket.copy()                                    

    p_coor = df_tmp[['x_coord', 'y_coord', 'z_coord']].values    
    p2l_dismap = spatial.distance.cdist(p_coor, center_coor, metric='euclidean')    
    p2l_dis = p2l_dismap.min(-1)                                                          
    df_tmp['pocket_flag'] = (p2l_dis < dis)                                               
    pocket_res_chain = []
    for res_i in df_tmp['residue_number'].unique():
        for chain_i in df_tmp['chain_id'].unique():
            df_sub = df_tmp[(df_tmp['residue_number'] == res_i) & (df_tmp['chain_id'] == chain_i)] 
            tmp_dis = df_sub['pocket_flag'].values                                       
            if any(tmp_dis):
                pocket_res_chain.append(f'{res_i},{chain_i}')                         

    df_pocket['pocket_flag'] = [True if str(df_pocket.loc[i, 'residue_number']) + ',' + str(        # 这个表达式感觉很难写啊
        df_pocket.loc[i, 'chain_id']) in pocket_res_chain else False                   
                                for i in range(len(df_pocket))]
    df_pocket = df_pocket[df_pocket['pocket_flag'] == True][df_pocket.columns[:-1]]   
    return df_pocket



# 格式转换
def parse_triplet(vals):
    if vals is None:
        return None
    return (float(vals[0]), float(vals[1]), float(vals[2]))



# 计算grid盒子大小（） 计算依据
def compute_box_bbox(coords, padding):
    """Return (bbox_center, box_size) — center is the bounding-box midpoint, not the atomic centroid."""
    arr = np.asarray(coords)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    center = tuple(float(v) for v in (lo + hi) / 2)
    size = tuple(float(v) for v in (hi - lo) + 2 * padding)
    return center, size

def box_size_to_npts(box_size):
    npts = []
    for size in box_size:
        count = int(math.ceil(size / GRID_SPACING))
        if count % 2 != 0:
            count += 1
        npts.append(count)
    return tuple(npts)



# 解析输出的文件内容
FINAL_SUMMARY_PREFIXES = (
    "USER    Run =",
    "USER    Cluster Rank =",
    "USER    Number of conformations in this cluster =",
    "USER    (1) Final Intermolecular Energy",
    "USER        vdW + Hbond + desolv Energy",
    "USER        Electrostatic Energy",
    "USER    (2) Final Total Internal Energy",
    "USER    (3) Torsional Free Energy",
    "USER    (4) Unbound System's Energy",
)


def parse_dlg_results(dlg_path):
    lines = Path(dlg_path).read_text(encoding="utf-8").splitlines()
    current_run = None
    run_energies = []
    final_summary = []
    in_final_model = False

    for line in lines:
        stripped = line.strip()
        if line.startswith("DOCKED: USER    Run ="):
            current_run = int(line.split("=", 1)[1].strip())
        elif current_run is not None and "Estimated Free Energy of Binding" in line and line.startswith("DOCKED: USER"):
            run_energies.append(f"Run {current_run}: {stripped}")
        elif line.startswith("MODEL"):
            in_final_model = True
        elif in_final_model and (stripped.startswith(FINAL_SUMMARY_PREFIXES) or
                                  ("Estimated Free Energy of Binding" in line and line.startswith("USER"))):
            final_summary.append(stripped)
            if stripped.startswith("USER    (4) Unbound System's Energy"):
                break

    summary = list(run_energies)
    if final_summary:
        summary.append("Best cluster summary:")
        summary.extend(final_summary)
    return summary






# 这两个函数好像没什么用
# def compute_centroid(coords):
#     arr = np.array(coords)
#     return tuple(arr.mean(axis=0))

# def compute_box_size(coords, padding):# -> tuple:
#     arr = np.asarray(coords)
#     extents = arr.max(axis=0) - arr.min(axis=0)
#     return tuple(map(float, extents + 2 * padding))
