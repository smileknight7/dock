from biopandas.pdb import PandasPdb
import pandas as pd 
from pathlib import Path
import scipy.spatial    
from rdkit import Chem
from run_mopac.config import STANDARD_PROTEIN_ATOMS
import re
from run_mopac.mopac_utils import _extract_first_three_floats
import numpy as np


def parse_pdb(pdb_path, include_hetatm=True):                                # pandas_PDB不能强行从pdbqt中读取
    '''returns a list of dicts with keys: chain, res_id, res_name, x, y, z, element, line'''
    pdb_path = Path(pdb_path)
    ppdb = PandasPdb().read_pdb(str(pdb_path))
    if include_hetatm:
        df_pdb = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]], ignore_index=True)
    else:   
        df_pdb = ppdb.df["ATOM"]
    return df_pdb



def select_pocket(df_protein, center_coor, dis, any_atom=False):   # any_atom=True,  center_coor---->ligand_pos
    '''df_protein 使用pandaspdb构建， center_coor是ligand的坐标'''
    
    if not any_atom:
        df_tmp = df_protein[df_protein['atom_name'] == 'CA'].copy()
    else:
        df_tmp = df_protein.copy()                                    

    p_coor = df_tmp[['x_coord', 'y_coord', 'z_coord']].values    
    p2l_dismap = scipy.spatial.distance.cdist(p_coor, center_coor, metric='euclidean')    
    p2l_dis = p2l_dismap.min(-1)                                                          
    df_tmp['pocket_flag'] = (p2l_dis < dis)                                               
    pocket_res_chain = set(                                                         # 获取到dist范围内的残基编号和链ID的组合
        zip(
            df_tmp.loc[df_tmp['pocket_flag'], 'residue_number'],
            df_tmp.loc[df_tmp['pocket_flag'], 'chain_id'],
        )
    )

    residue_keys = zip(df_protein['residue_number'], df_protein['chain_id'])
    pocket_flag = [key in pocket_res_chain for key in residue_keys]                 # 得到布尔值
    return df_protein.loc[pocket_flag].copy()                                        # .loc接受两个参数，第一个是布尔索引取行裁剪df，第二个参数是列名字得到列内容



def write_df2_pdb(df_pocket, output_path):
    '''edit pandas_df to pdb format'''
    output_path = Path(output_path)
    out = PandasPdb()
    out.df['ATOM'] = df_pocket[df_pocket['record_name'] == 'ATOM'].reset_index(drop=True)
    out.df['HETATM'] = df_pocket[df_pocket['record_name'] == 'HETATM'].reset_index(drop=True)
    out.to_pdb(str(output_path), records=['ATOM', 'HETATM'])

    # BioPandas 0.5.1 writes columns 67-73 for blank_4, although the PDB
    # specification reserves only columns 67-72. This shifts segment_id,
    # element_symbol, and charge one column to the right. Apply the workaround
    # only when the installed formatter still has that seven-character field.
    # 这里说是pandasPDB导出pdb_file的时候会识别错号，向右多识别一位（这里进行修正一下）         
    
    from biopandas.pdb.engines import pdb_records                              

    blank_4_spec = next(
        field for field in pdb_records["ATOM"] if field["id"] == "blank_4"
    )
    if len(blank_4_spec["strf"]("")) == 7:
        lines = output_path.read_text().splitlines()
        corrected_lines = []
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                line = (line[:72] + line[73:]).ljust(80)
            corrected_lines.append(line)
        output_path.write_text("\n".join(corrected_lines) + "\n")

    # 返回 mol 对象
    return Chem.MolFromPDBFile(str(output_path), removeHs=False)




def read_rdkit_mol(mol):
    if mol.endswith('pdb'):
        mol = Chem.MolFromPDBFile(mol)
    elif mol.endswith('mol'):
        mol = Chem.MolFromMolFile(mol)
    elif mol.endswith('mol2'):
        mol = Chem.MolFromMol2File(mol)
    elif mol.endswith('sdf'):
        SD = Chem.SDMolSupplier(mol)
        mol = [x for x in SD][0]
    else:
        mol = Chem.MolFromSmiles(mol)
    return mol



def extract_ligand(df_cov_ligand):
    """从共价配体文件中提取纯配体（排除标准蛋白质原子）"""
    ligand_mask = ~df_cov_ligand['atom_name'].str.strip().isin(STANDARD_PROTEIN_ATOMS)
    df_ligand = df_cov_ligand[ligand_mask].copy()

    return df_ligand

def merge_pocket(df_cov_ligand, df_rigid_pocket):
    """拼接flex_pocket和rigid_pocket，得到完整的pocket"""
    pocket_mask = df_cov_ligand['atom_name'].str.strip().isin(STANDARD_PROTEIN_ATOMS)
    df_flex_pocket = df_cov_ligand[pocket_mask].copy()
    df_pocket = pd.concat([df_flex_pocket, df_rigid_pocket], ignore_index=True)
    return df_pocket




def extract_from_pdb_block(block_file: str | Path) -> tuple[list[str], np.ndarray]:
    element = []
    coor = []
    with open(block_file) as handle:
        for line in handle:  # 不跳过任何行！
            line = line.strip()
            if len(line) < 5:
                continue

            # Block lines start with labels such as ``C(ATOM0`` or
            # ``Cl(ATOM12``.  Keep the complete element symbol and discard
            # the atom annotation before converting to plain MOPAC XYZ.
            match = re.match(r"([A-Za-z]+)\(", line)
            if match is None:
                continue

            elem = match.group(1)
            xyz = _extract_first_three_floats(line)
            if len(xyz) != 3:
                continue
            element.append(elem)
            coor.append(xyz)
    return element, np.asarray(coor, dtype=float)
