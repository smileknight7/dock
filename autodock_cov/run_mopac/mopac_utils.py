from pathlib import Path
from rdkit import Chem
import scipy.spatial
from biopandas.pdb import PandasPdb
import pandas as pd 
import numpy as np  
import re
from typing import Iterable
from rdkit.Chem import AllChem
import re



def _extract_first_three_floats(text: str) -> list[float]:
    # Only match numbers with decimal points to avoid matching +0/+1 optimization flags
    nums = re.findall(r"[-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?", text)
    return [float(num.replace("D", "E").replace("d", "e")) for num in nums[:3]]


def optimize_h_only(mol, conf_id=0, max_iters=200):
    # 只对H部分进行优化
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError("Some atoms do not have MMFF parameters")

    mmff_props = AllChem.MMFFGetMoleculeProperties(
        mol,
        mmffVariant="MMFF94",
    )

    ff = AllChem.MMFFGetMoleculeForceField(
        mol,
        mmff_props,
        confId=conf_id,
        ignoreInterfragInteractions=False,
    )

    # 固定所有重原子，只允许 H 移动
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            ff.AddFixedPoint(atom.GetIdx())

    ff.Initialize()
    status = ff.Minimize(maxIts=max_iters)

    return status



def build_pdb_block_from_df(df_pdb, opt_flag) -> str:
    '''构建成pdb文件用于将df转pdb????但是我不是写了一个df -->pdb的函数了吗'''
    lines = []
    for i, row in df_pdb.iterrows():
        # import ipdb; ipdb.set_trace()
        element = str(row["element_symbol"]).strip()
        serial = str(i)
        atom_name = str(row["atom_name"]).strip().ljust(4)
        res_name = str(row["residue_name"]).strip()
        chain = str(row["chain_id"]).strip().rjust(2)
        res_id = str(row["residue_number"]).rjust(4)

        x, y, z = (
            f"{float(coord):.8f}".rjust(13)
            for coord in row[["x_coord", "y_coord", "z_coord"]]
        )
        flag = int(opt_flag)
        lines.append(
            f"{element}(ATOM{serial}  {atom_name}{res_name}{chain}{res_id})"
            f"{x} +{flag}{y} +{flag}{z} +{flag}"
        )
    return "\n".join(lines)


def elem_coor_to_xyz(
    element: Iterable[str],
    coor: np.ndarray,
    only_opt_H: bool = False,
    opt_flag: bool = False,
) -> str:
    lines = []
    for elem, xyz in zip(element, coor):
        if opt_flag:                       # 全部优化
            flag = 1
        elif only_opt_H:                   # 只优化 H
            flag = 1 if elem == "H" else 0
        else:                              # 全部冻结
            flag = 0

        x, y, z = (f"{float(v):.6f}".rjust(11) for v in xyz)
        lines.append(
            f"{str(elem).ljust(4)}{x} {flag}{y} {flag}{z} {flag}"
        )
    return "\n".join(lines)


def mol_to_pymopac_xyz(mol) -> str:
    '''convert rdkit mol to pymopac xyz format'''
    lines = Chem.MolToXYZBlock(mol).splitlines()
    if lines and lines[0].strip().isdigit():
        lines = lines[2:]
    return "\n".join(lines)


def assign_coor_to_mol(mol, coor: np.ndarray):
    '''给 mol 套上一组新坐标（替换全部 conformer），返回新 mol。'''
    coor = np.asarray(coor, dtype=float)
    if coor.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(
            f"Coordinate shape {coor.shape} does not match {mol.GetNumAtoms()} atoms."
        )
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, xyz in enumerate(coor):
        conf.SetAtomPosition(idx, tuple(map(float, xyz)))
    mol.AddConformer(conf, assignId=True)
    return mol


def extract_traj(mopac_keywords: str, aux_file) -> dict:
    '''解析 MOPAC .aux 的优化轨迹，返回 coor/grad/energy/grad_norm。

    需要 PRNT=2。coor=(步数,原子数,3) 含全部原子；grad=(步数,可动原子数,3) 只含被优化原子。
    '''
    assert 'PRNT=2' in mopac_keywords, mopac_keywords

    blocks, block, is_traj = [], [], False
    for line in open(aux_file, 'r').readlines():
        if line.startswith(' HEAT_OF_FORM'):
            if is_traj and len(block) > 0:
                blocks.append(block)
            is_traj = True
            block = []
        if is_traj:
            block.append(line)
    blocks.append(block)

    coor_list, grad_list, energy_list, grad_norm_list = [], [], [], []
    for block in blocks:
        is_coor = is_grad = False
        coor, grad = [], []
        for line in block:
            if line.startswith(' HEAT_OF_FORM'):
                energy_list.append(float(line.split('=')[1][:-1].replace('D', 'e')))
            elif line.startswith(' GRADIENT_NORM'):
                grad_norm_list.append(float(line.split('=')[1][:-1].replace('D', 'e')))
            elif line.startswith(' ATOM_X'):
                is_coor = True
                continue
            elif line.startswith(' GRADIENTS'):
                is_grad = True
                continue
            if is_coor:
                nums = re.findall(r'[-+]?\d*\.\d+', line)
                if len(nums) != 3:
                    is_coor = False
                else:
                    coor.append([float(n) for n in nums])
            if is_grad:
                if re.findall(r'[A-Za-z]+', line):
                    is_grad = False
                else:
                    grad.extend([float(n) for n in re.findall(r'[-+]?\d*\.\d+', line)])
        coor_list.append(np.array(coor))
        grad_list.append(np.array(grad).reshape(-1, 3))

    return dict(
        coor=np.stack(coor_list, 0),
        grad=np.stack(grad_list, 0),
        energy=np.array(energy_list),
        grad_norm=np.array(grad_norm_list),
    )








