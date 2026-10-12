from openbabel import openbabel
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit import Chem
import numpy as np
from config import AA_ATOMS
import re
import pandas as pd
from pathlib import Path


sym2idx = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53
}


# modified from https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/main/rf2aa/data/parsers.py#L744
def parse_mol(filename, filetype="pdb", string=False, remove_H=True, generate_conformer: bool = False):
    """得到pocket的mol表示，和reaction的atom_idx

    Parameters
    ----------
    filename : str                                              # 使用openbabel文件路径要用str
    filetype : str
    string : bool
        If True, `filename` is a string containing the molecule data.
    remove_H : bool
        Whether to remove hydrogen atoms.

    Returns
    -------
    obmol: OBMol
        openbabel molecule object representing the pocket
    sdf_string: str
        converted sdf file string
    atom_types: List[int] (N_atoms, )
    atom_coords: List[List[int]] (N_atoms, 3) float
        Atom coordinates
    bond_table: List[List[int]] (N_bonds, 2)
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(filetype)
    obmol = openbabel.OBMol()
    if string:
        obConversion.ReadString(obmol,filename)
    # elif filetype=='sdf':
    #     molstring = clean_sdffile(filename)
    #     obConversion.ReadString(obmol,molstring)
    else:
        obConversion.ReadFile(obmol,filename)
    if generate_conformer:
        builder = openbabel.OBBuilder()
        builder.Build(obmol)
        ff = openbabel.OBForceField.FindForceField("mmff94")
        did_setup = ff.Setup(obmol)
        if did_setup:
            ff.FastRotorSearch()
            ff.GetCoordinates(obmol)
        else:
            raise ValueError(f"Failed to generate 3D coordinates for molecule {filename}.")
    if remove_H:
        obmol.DeleteHydrogens()
        # the above sometimes fails to get all the hydrogens
        i = 1
        while i < obmol.NumAtoms()+1:
            if obmol.GetAtom(i).GetAtomicNum()==1:
                obmol.DeleteAtom(obmol.GetAtom(i))
            else:
                i += 1
    atom_types = [obmol.GetAtom(i).GetAtomicNum()
                 for i in range(1, obmol.NumAtoms()+1)]
    atom_coords = [[obmol.GetAtom(i).x(),obmol.GetAtom(i).y(), obmol.GetAtom(i).z()] 
                                for i in range(1, obmol.NumAtoms()+1)]# (natoms, 3)
    bond_table = [[b.GetBeginAtomIdx()-1, b.GetEndAtomIdx()-1, b.GetBondOrder()] for b in openbabel.OBMolBondIter(obmol)]
    obConversion.SetOutFormat("sdf")
    sdf_string = obConversion.WriteString(obmol)
    return obmol, sdf_string, atom_types, atom_coords, bond_table




def _find_pocket_bond_idx(pdb_filename, pocket_bond_list, ignore_H=True):
    '''返回pocket中对应covalent bond的idx'''
    with open(pdb_filename, 'r') as f:
        lines = f.readlines()
    ter_cnt = 0
    H_cnt = 0
    D_cnt = 0
    for i, line in enumerate(lines):
        if line.startswith('END'):                                      # 非标准pdb文件字段
            continue
        if line.startswith('TER'):
            ter_cnt += 1
            continue
        name = line[12:16].strip(); resName = line[17:20].strip()
        chainID = line[21]; resSeq = line[22:26].strip()
        if ignore_H and name.startswith('H'):
            H_cnt += 1
            continue
        if name.startswith('D'):
            D_cnt += 1
            continue
        if [name,resName,chainID,resSeq] == pocket_bond_list:
            return i - ter_cnt - H_cnt - D_cnt
    return None

def assign_substruct_coords(mol, match_indices, new_coords):
    '''将子结构的新坐标应用到原来的mol中'''
    mol.RemoveAllConformers()
    AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()
    
    for i,mol_idx in enumerate(match_indices):
        x,y,z = new_coords[i]
        conf.SetAtomPosition(mol_idx,Point3D(x,y,z))


def generate_conformation(mol, seed=-1):
    '''为ligand生成合理的3D构象;seed 固定可复现,-1 为随机(默认,保持原行为)'''
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    ps.randomSeed = seed
    try:
        rid = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    except:
        mol.Compute2DCoords()
    mol = Chem.RemoveHs(mol)
    return mol


def replace_file_content(file_path, old_str, new_str):
    # Read in the file
    with open(file_path, 'r') as fp:
        filedata = fp.read()

    # Replace the target string
    filedata = filedata.replace(old_str, new_str)

    # Write the file out again
    with open(file_path, 'w') as fp:
        fp.write(filedata)


def clean_sym(sym):
    sym = sym.strip().title()
    sym = sym.replace('+','')
    sym = sym.replace('-','')
    sym = ''.join(filter(lambda x: not x.isdigit(), sym)) # drop all number
    return sym


def _extract_ligand_pdb_block_from_prediction(pdb_filename):

    ligand_lines = []
    fallback_lines = []
    with open(pdb_filename, 'r') as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in AA_ATOMS:
            # if atom_name.endswith('X'):
                ligand_lines.append(line)
            fallback_lines.append(line)

    selected_lines = ligand_lines if ligand_lines else fallback_lines
    if not selected_lines:
        raise ValueError(f'Failed to extract ligand atoms from {pdb_filename}')
    return ''.join(selected_lines) + 'END\n'


def calculate_rmsd(P, Q):
    """Calculate the RMSD between two aligned point clouds."""
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))


def _calc_template_order_rmsd(template_mol, pred_mol, matches):
    template_conf = template_mol.GetConformer()
    pred_conf = pred_mol.GetConformer()
    best_rmsd = float('inf')
    best_match = None
    for match in matches:
        template_coords = []
        pred_coords = []
        for template_idx, pred_idx in enumerate(match):
            template_pos = template_conf.GetAtomPosition(template_idx)
            pred_pos = pred_conf.GetAtomPosition(pred_idx)
            template_coords.append([template_pos.x, template_pos.y, template_pos.z])
            pred_coords.append([pred_pos.x, pred_pos.y, pred_pos.z])
        template_coords = np.array(template_coords)
        pred_coords = np.array(pred_coords)
        rmsd = calculate_rmsd(template_coords, pred_coords)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_match = match
    return best_rmsd, best_match





def parse_dlg_energies(dlg_file):
    """解析 AutoDock4 .dlg 里每个 DOCKED 构象的结合能(kcal/mol,越负越好)。

    返回 {pose_idx: energy};pose_idx 按 DOCKED 块在文件中的出现顺序(= run 号),
    与 write_conformations_from_dlg.py 写出的 res_ligand_pose_{i} 编号一致。
    """
    e_re = re.compile(r'DOCKED:\s*USER\s+Estimated Free Energy of Binding\s*=\s*([-+]?\d*\.?\d+)')
    energies = {}
    with open(dlg_file) as f:
        for line in f:
            m = e_re.search(line)
            if m:
                energies[len(energies) + 1] = float(m.group(1))
    return energies


def parse_dlg_energies_multi(dock_result_i_dir):
    """聚合一个样本下 conf_*/autodock_output.dlg 的结合能,按 conf 顺序拼成连续 pose_idx。

    与 get_pose 的 pose 重编号严格一致:conf_1 → 1..n1, conf_2 → n1+1..n1+n2,以此类推。
    返回 {pose_idx: energy}。
    """
    dock_result_i_dir = Path(dock_result_i_dir)
    conf_dirs = sorted(dock_result_i_dir.glob('conf_*'), key=lambda p: int(p.name.split('_')[1]))
    energies = {}
    offset = 0
    for conf_dir in conf_dirs:
        e = parse_dlg_energies(conf_dir / 'autodock_output.dlg')   # {1..n: energy}
        for idx, en in e.items():
            energies[offset + idx] = en
        offset += len(e)
    return energies


def parse_epdb_energy(dlg_file):
    """解析 AutoDock4 epdb 单点评分输出里的结合能(kcal/mol,越负越好)。

    epdb 行形如:
        epdb: USER    Estimated Free Energy of Binding    =   -7.42 kcal/mol
    返回 float;未找到返回 None。注意能量可能是科学计数法——碰撞严重的输入构象
    会给出 +5.3e+05 之类的天文正值,这说明喂进去的 pose 本身有原子重叠。
    """
    e_re = re.compile(
        r'epdb:\s*USER\s+Estimated Free Energy of Binding\s*=\s*'
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    )
    with open(dlg_file) as f:
        for line in f:
            m = e_re.search(line)
            if m:
                return float(m.group(1))
    return None
