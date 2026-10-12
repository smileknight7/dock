from __future__ import annotations

from importlib import import_module
import re
import shutil
from pathlib import Path
from typing import Iterable
from mopac.mopac_utils import build_pocket_block_from_df, write_pocket_pdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import pymopac

RDLogger.DisableLog("rdApp.*")


def optimize_h_only(mol, conf_id=0, max_iters=200):
    """只对 H 原子进行 MMFF 优化，固定所有重原子"""
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError("Some atoms do not have MMFF parameters")

    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    ff = AllChem.MMFFGetMoleculeForceField(
        mol, mmff_props, confId=conf_id, ignoreInterfragInteractions=False
    )

    # 固定所有重原子，只允许 H 移动
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            ff.AddFixedPoint(atom.GetIdx())

    ff.Initialize()
    return ff.Minimize(maxIts=max_iters)


try:
    RD_DETERMINE_BONDS = import_module("rdkit.Chem.rdDetermineBonds")
except ImportError:
    RD_DETERMINE_BONDS = None


# General helpers
def reset_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Old alias; keep this while older callers still use delmkdir.
delmkdir = reset_dir


def load_idx_list(path: str | Path) -> list[str]:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def print_dic_data(data: dict) -> None:
    for key, value in data.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value).__name__}")


def debug() -> None:
    return None


# RDKit coordinate helpers
def assign_coor_to_mol(mol, coor: np.ndarray):
    coor = np.asarray(coor, dtype=float)
    if coor.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(
            f"Coordinate shape {coor.shape} does not match "
            f"{mol.GetNumAtoms()} atoms."
        )

    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, xyz in enumerate(coor):
        conf.SetAtomPosition(idx, tuple(map(float, xyz)))
    mol.AddConformer(conf, assignId=True)
    return mol


def get_mol_posi(mol) -> np.ndarray:
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer coordinates.")
    conf = mol.GetConformer()
    return np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )


def mol_to_pymopac_xyz(mol) -> str:
    lines = Chem.MolToXYZBlock(mol).splitlines()
    if lines and lines[0].strip().isdigit():
        lines = lines[2:]
    return "\n".join(lines)


def smi_assign_coor_to_pdb(                             # 感觉这里从mol文件构建更好
    mol,
    coor: np.ndarray,
    output_path: str | Path,
) -> str:
    if mol is None:
        raise ValueError("error need ligand_mol")
    AllChem.Compute2DCoords(mol)
    mol = assign_coor_to_mol(mol, coor)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToPDBFile(mol, str(output_path))
    return str(output_path)


def xyz2mol(
    element: Iterable[str],
    coor: np.ndarray,
    add_bond: bool = True,
) -> Chem.Mol:
    mol = Chem.RWMol()
    element = list(element)
    coor = np.asarray(coor, dtype=float)

    for elem in element:
        mol.AddAtom(Chem.Atom(str(elem)))

    conf = Chem.Conformer(len(element))
    for idx, xyz in enumerate(coor):
        conf.SetAtomPosition(idx, tuple(map(float, xyz)))
    mol.AddConformer(conf)
    mol = mol.GetMol()

    if add_bond:
        if RD_DETERMINE_BONDS is None:
            raise ImportError("rdkit.Chem.rdDetermineBonds is not available.")
        RD_DETERMINE_BONDS.DetermineConnectivity(
            mol,
            useHueckel=False,
            useVdw=True,
        )

    return mol


# MOPAC output parsing
def _extract_first_three_floats(text: str) -> list[float]:
    # Only match numbers with decimal points to avoid matching +0/+1 optimization flags
    nums = re.findall(r"[-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?", text)
    return [float(num.replace("D", "E").replace("d", "e")) for num in nums[:3]]


def extract_charged(aux_file: str | Path) -> tuple[list[str], np.ndarray]:
    element = []
    coor = []
    with open(aux_file) as handle:
        for line in handle.readlines()[3:]:
            if len(line.strip()) < 5:
                continue

            elem = line.split()[0]
            xyz = _extract_first_three_floats(line)
            if len(xyz) != 3:
                continue
            element.append(elem)
            coor.append(xyz)

    coor = np.array(coor, dtype=float)
    if len(element) != len(coor):
        raise ValueError(f"Element/coordinate count mismatch in {aux_file}")
    return element, coor


def extract_pocket_H(aux_file: str | Path) -> tuple[list[str], np.ndarray]:
    element = []
    coor = []
    with open(aux_file) as handle:
        for line in handle.readlines()[3:]:
            if len(line.strip()) < 5:
                continue

            # Handle lines with leading whitespace
            match = re.findall(r"^\s*([A-Za-z]+)\(", line)
            if match:
                elem = match[0]
                coord_text = line.split(")", 1)[1]
            else:
                parts = line.split(maxsplit=1)
                if not parts:
                    continue
                elem = parts[0]
                coord_text = parts[1] if len(parts) > 1 else ""

            xyz = _extract_first_three_floats(coord_text)
            if len(xyz) != 3:
                continue
            element.append(elem)
            coor.append(xyz)

    coor = np.array(coor, dtype=float)
    if len(element) != len(coor):
        raise ValueError(f"Element/coordinate count mismatch in {aux_file}")
    return element, coor


def extract_traj(mopac_keywords: str, aux_file: str | Path) -> dict[str, np.ndarray]:
    if "PRNT=2" not in mopac_keywords:
        raise ValueError(f"PRNT=2 is required to extract trajectory: {mopac_keywords}")

    blocks = []
    block = []
    is_traj = False
    with open(aux_file) as handle:
        for line in handle:
            if line.startswith(" HEAT_OF_FORM"):
                if is_traj and block:
                    blocks.append(block)
                is_traj = True
                block = []

            if is_traj:
                block.append(line)
    if block:
        blocks.append(block)

    coor_list = []
    grad_list = []
    energy_list = []
    grad_norm_list = []
    for block in blocks:
        is_coor = False
        is_grad = False
        coor = []
        grad = []

        for line in block:
            if line.startswith(" HEAT_OF_FORM"):
                heat = line.split("=")[1]
                energy_list.append(float(heat[:-1].replace("D", "e")))
            elif line.startswith(" GRADIENT_NORM"):
                grad_norm = line.split("=")[1]
                grad_norm_list.append(float(grad_norm[:-1].replace("D", "e")))
            elif line.startswith(" ATOM_X"):
                is_coor = True
                continue
            elif line.startswith(" GRADIENTS"):
                is_grad = True
                continue

            if is_coor:
                nums = re.findall(r"[-+]?\d*\.\d+", line)
                if len(nums) != 3:
                    is_coor = False
                else:
                    coor.append([float(nums[0]), float(nums[1]), float(nums[2])])

            if is_grad:
                non_nums = re.findall(r"[A-Za-z]+", line)
                if non_nums:
                    is_grad = False
                else:
                    nums = re.findall(r"[-+]?\d*\.\d+", line)
                    grad.extend([float(num) for num in nums])

        coor_list.append(np.array(coor, dtype=float))
        grad_list.append(np.array(grad, dtype=float).reshape(-1, 3))

    return {
        "coor": np.stack(coor_list, 0),
        "grad": np.stack(grad_list, 0),
        "energy": np.array(energy_list, dtype=float),
        "grad_norm": np.array(grad_norm_list, dtype=float),
    }


# MOPAC input and execution helpers
def elem_coor_to_xyz(
    element: Iterable[str],
    coor: np.ndarray,
    only_opt_H: bool = False,
) -> str:
    lines = []
    for elem, xyz in zip(element, coor):
        opt_flag = 1
        if only_opt_H:
            opt_flag = 1 if elem == "H" else 0

        x, y, z = (f"{float(v):.6f}".rjust(11) for v in xyz)
        lines.append(
            f"{str(elem).ljust(4)}{x} {opt_flag}{y} {opt_flag}{z} {opt_flag}"
        )

    return "\n".join(lines)


def run_comp_mopac(
    mol,
    pocket_mol,
    pocket_block: str,
    mopac_kw: str,
    rdkit_addh: bool = False,
    tmp_path: str | Path = "./tmp",
) -> dict[str, np.ndarray]:
    tmp_path = reset_dir(tmp_path).resolve()

    # Step 1: Add hydrogens to pocket using MOPAC ADD-H
    if not rdkit_addh:
        pocket_H_path = tmp_path / "pocket_H"
        pocket_H_path.mkdir(parents=True, exist_ok=True)

        if mol is None:
            raise ValueError(
                "Ligand molecule is None; cannot initialize the MOPAC ADD-H job."
            )

        # pymopac parses ``geometry`` immediately in MopacInput.__init__.
        # Passing "\n" makes RDKit parse an empty SMILES and then call
        # MolToXYZBlock(None).  Initialize with a valid molecule, then clear
        # xyz because GEO_DAT supplies the actual pocket geometry.
        runner = pymopac.MopacInput(
            mol,
            model='GEO_DAT="pocket.txt" ADD-H SITE=(SALT) GEO-OK',
            comment="# pocket add H & ionize",
            path=str(pocket_H_path),
            addHs=False,
            preopt=False,
            verbose=True,
            stream=False,
            aux=False,
        )
        (pocket_H_path / "pocket.txt").write_text(pocket_block)
        runner.xyz = "\n"
        runner.run()
        element, coor = extract_pocket_H(pocket_H_path / "pymopac.arc")
        pocket_charged_block = elem_coor_to_xyz(element, coor, only_opt_H=True)
        
    
    else:
        # RDKit 加氢分支（非共价）
        pocket_addh_mol = Chem.AddHs(pocket_mol, addCoords=True)
        optimize_h_only(pocket_addh_mol, max_iters=200)

        # 提取元素和坐标，转换为 MOPAC 格式
        element = [atom.GetSymbol() for atom in pocket_addh_mol.GetAtoms()]
        coor = get_mol_posi(pocket_addh_mol)
        pocket_charged_block = elem_coor_to_xyz(element, coor, only_opt_H=True)   



    # Step 2: Ionize ligand using MOPAC SITE=(SALT)
    # Use mol directly if it already has hydrogens, otherwise add them
    if mol.GetNumAtoms() > 0 and any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        mol_H = mol
    else:
        mol_H = Chem.AddHs(mol, addCoords=True)

    charged_path = tmp_path / "charged"
    charged_path.mkdir(parents=True, exist_ok=True)
    pymopac.MopacInput(
        mol_to_pymopac_xyz(mol_H),
        model="SITE=(SALT)",
        comment="# ligand add H & ionize",
        path=str(charged_path),
        addHs=False,
        preopt=False,
        verbose=True,
        stream=False,
        aux=False,
    ).run()

    element, coor = extract_charged(charged_path / "pymopac.arc")
    mol_charged_block = elem_coor_to_xyz(element, coor)

    complex_block = mol_charged_block + "\n" + pocket_charged_block

    # Step 3: Run production MOPAC calculation on ligand + pocket complex
    product_run_path = tmp_path / "product_run"
    product_run_path.mkdir(parents=True, exist_ok=True)

    # 直接写 MOPAC 输入文件（绕过 pymopac 的输入解析问题）
    mop_content = f"""{mopac_kw}
# Product run

{complex_block}
"""
    mop_file = product_run_path / "pymopac.mop"
    mop_file.write_text(mop_content)

    # 直接运行 mopac
    import subprocess
    subprocess.run(["mopac", str(mop_file)], cwd=str(product_run_path), check=True)

    return extract_traj(mopac_kw, product_run_path / "pymopac.aux")
