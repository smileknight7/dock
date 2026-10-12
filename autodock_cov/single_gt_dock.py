from pathlib import Path
from config import (
    run_prepare_receptor4,
    run_prepare_flex_receptor4,
    run_prepare_dpf4,
    run_prepare_gpf4,
    run_autogrid4,
    run_autodock4,
)
from utils.cov_utils import replace_file_content
from rdkit import Chem
import pandas as pd
import os


def _split_bond_res(bond_res):
    chain_id, residue = bond_res.split(':', 1)
    return chain_id, residue[:3], residue[3:]


def _pdb_element(atom_name, line):
    element = line[76:78].strip()
    if element:
        return element
    return ''.join(c for c in atom_name if c.isalpha())[:1].upper()


def _read_residue_atoms(pocket_file, bond_res):
    chain_id, res_name, res_seq = _split_bond_res(bond_res)
    atoms = []
    for line in Path(pocket_file).read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        atom_name = line[12:16].strip()
        if (
            line[17:20].strip() == res_name
            and line[21].strip() == chain_id
            and line[22:26].strip() == res_seq
            and not atom_name.startswith(('H', 'D'))
        ):
            atoms.append({
                'name': atom_name,
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
                'element': _pdb_element(atom_name, line),
            })
    if not atoms:
        raise ValueError(f'Cannot find residue {bond_res} in {pocket_file}')
    return atoms


def _nearest_atom(coord, residue_atoms, excluded=()):
    excluded = set(excluded)
    best_atom = None
    best_dist2 = float('inf')
    for atom in residue_atoms:
        if atom['name'] in excluded:
            continue
        dist2 = (
            (coord[0] - atom['x']) ** 2
            + (coord[1] - atom['y']) ** 2
            + (coord[2] - atom['z']) ** 2
        )
        if dist2 < best_dist2:
            best_atom = atom
            best_dist2 = dist2
    if best_atom is None:
        raise ValueError('Cannot match GT SDF anchor atom to pocket residue atom')
    return best_atom


def _pdb_atom_line(serial, atom_name, res_name, chain_id, res_seq, x, y, z, element):
    return (
        f'HETATM{serial:5d} {atom_name:>4} {res_name:>3} {chain_id:1}{res_seq:>4}'
        f'    {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}  '
    )


def _residue_bond_cutoff(element_a, element_b):
    if 'S' in {element_a.upper(), element_b.upper()}:
        return 2.2
    return 1.95


def write_gt_covalent_pdb(sdf_file, pocket_file, output_pdb, bond_res, ligand_indices):
    """Build the prepare_flexreceptor4 input from an already aligned GT SDF."""
    mol = Chem.SDMolSupplier(str(sdf_file), removeHs=False)[0]
    if mol is None:
        raise ValueError(f'Cannot read {sdf_file}')

    chain_id, res_name, res_seq = _split_bond_res(bond_res)
    residue_atoms = _read_residue_atoms(pocket_file, bond_res)
    second_aa_idx, aa_idx = [int(i) - 1 for i in ligand_indices.split(',')]
    conf = mol.GetConformer()

    aa_pos = conf.GetAtomPosition(aa_idx)
    bond_atom = _nearest_atom((aa_pos.x, aa_pos.y, aa_pos.z), residue_atoms)
    second_pos = conf.GetAtomPosition(second_aa_idx)
    second_atom = _nearest_atom(
        (second_pos.x, second_pos.y, second_pos.z),
        residue_atoms,
        excluded=(bond_atom['name'],),
    )

    atom_lines = []
    residue_serials = {}
    element_counts = {}
    serial = 1
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        element = atom.GetSymbol()
        if idx == aa_idx:
            atom_name = bond_atom['name']
        elif idx == second_aa_idx:
            atom_name = second_atom['name']
        else:
            element_counts[element] = element_counts.get(element, 0) + 1
            atom_name = f'{element}{element_counts[element]}'[:4]

        pos = conf.GetAtomPosition(idx)
        atom_lines.append(_pdb_atom_line(
            serial, atom_name, res_name, chain_id, res_seq,
            pos.x, pos.y, pos.z, element,
        ))
        if atom_name in {bond_atom['name'], second_atom['name']}:
            residue_serials[atom_name] = {
                'serial': serial,
                'x': pos.x,
                'y': pos.y,
                'z': pos.z,
                'element': element,
            }
        serial += 1

    for residue_atom in residue_atoms:
        if residue_atom['name'] in residue_serials:
            continue
        atom_lines.append(_pdb_atom_line(
            serial,
            residue_atom['name'],
            res_name,
            chain_id,
            res_seq,
            residue_atom['x'],
            residue_atom['y'],
            residue_atom['z'],
            residue_atom['element'],
        ))
        residue_serials[residue_atom['name']] = {'serial': serial, **residue_atom}
        serial += 1

    bond_pairs = set()
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx() + 1
        b = bond.GetEndAtomIdx() + 1
        bond_pairs.add(tuple(sorted((a, b))))

    residue_items = list(residue_serials.values())
    for i, atom_a in enumerate(residue_items):
        for atom_b in residue_items[i + 1:]:
            dist2 = (
                (atom_a['x'] - atom_b['x']) ** 2
                + (atom_a['y'] - atom_b['y']) ** 2
                + (atom_a['z'] - atom_b['z']) ** 2
            )
            cutoff = _residue_bond_cutoff(atom_a['element'], atom_b['element'])
            if 0.8 ** 2 <= dist2 <= cutoff ** 2:
                bond_pairs.add(tuple(sorted((atom_a['serial'], atom_b['serial']))))

    conect = {}
    for a, b in sorted(bond_pairs):
        conect.setdefault(a, []).append(b)
        conect.setdefault(b, []).append(a)
    conect_lines = [
        f'CONECT{atom_serial:5d}' + ''.join(f'{neighbor:5d}' for neighbor in neighbors)
        for atom_serial, neighbors in sorted(conect.items())
    ]

    Path(output_pdb).write_text('\n'.join(atom_lines + conect_lines + ['END']) + '\n')


def single_gt_dock(pdb_id, bond_res, work_dir, data_dir):
    data_i_dir = Path(data_dir/pdb_id)
    work_i_dir = Path(work_dir/pdb_id)
    os.makedirs(work_i_dir, exist_ok=True)

    with open(data_i_dir/'ligand_indices.txt', 'r') as f:
        ligand_indices = f.read().strip()
    # 预计算好的 grid box(中心=共价原子 SG;第一行 gridcenter,第二行 npts)
    grid_center, grid_npts = (data_i_dir/'grid_box.txt').read_text().splitlines()

    write_gt_covalent_pdb(
        data_i_dir/'autodock_gt.sdf',
        data_i_dir/'pocket.pdb',
        work_i_dir/'ligand_covalent_gt.pdb',
        bond_res,
        ligand_indices,
    )

    run_prepare_receptor4(['-r', f'{data_i_dir}/pocket.pdb', '-o', f'{work_i_dir}/pocket.pdbqt'], cwd=work_i_dir)
    run_prepare_receptor4(['-r', f'{work_i_dir}/ligand_covalent_gt.pdb'], cwd=work_i_dir)

    run_prepare_flex_receptor4(['-r', f'{work_i_dir}/pocket.pdbqt', '-s', f'pocket:{bond_res}'], cwd=work_i_dir)
    run_prepare_flex_receptor4(['-r', f'{work_i_dir}/ligand_covalent_gt.pdbqt', '-s', f'ligand_covalent_gt:{bond_res}'], cwd=work_i_dir)

    run_prepare_dpf4(['-r', 'pocket_rigid.pdbqt',
                        '-x', 'ligand_covalent_gt_flex.pdbqt',
                        '-l',  'ligand_covalent_gt_flex.pdbqt',
                        '-o', 'autodock_config.dpf',
                        '-p', 'move=empty',
                        '-p', 'ga_run=50',
                        '-p', 'ga_num_evals=2500000',
                        '-p', 'ga_mutation_rate=0.02',
                        '-p', 'ga_pop_size=150'], cwd=work_i_dir)
    run_prepare_gpf4(['-r', 'pocket_rigid.pdbqt',
                        '-l',  'ligand_covalent_gt_flex.pdbqt',
                        '-p', f'gridcenter={grid_center}',
                        '-p', f'npts={grid_npts}',
                        '-o', 'autogrid_config.gpf'], cwd=work_i_dir)

    (work_i_dir / 'empty').touch()
    replace_file_content(work_i_dir/'autogrid_config.gpf', 'unbound_model bound', 'unbound_energy 0.0')
    replace_file_content(work_i_dir/'autodock_config.dpf', 'unbound_model extended', 'unbound_energy 0.0')

    run_autogrid4(work_i_dir/'autogrid_config.gpf', work_i_dir/'autogrid_output.glg')
    run_autodock4(work_i_dir/'autodock_config.dpf', work_i_dir/'autodock_output.dlg')


if __name__ == "__main__":
    data_dir = Path('/data3/lpw/dock_workdir/autodock_cov/preparation')
    work_dir = Path('/data3/lpw/dock_workdir/autodock_cov/result_gt')
    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')
    os.makedirs(work_dir, exist_ok=True)

    df_info = pd.read_csv(info_file)
    pdb_id = '4HAX' # '4HAX' '4HCV'
    bond_info = df_info[df_info['pdb_id'] == pdb_id]['bond'].values[0]
    name, resName, chainID, resSeq = bond_info.split('-')[0].split(' ')
    bond_res = f'{chainID}:{resName}{resSeq}'
    single_gt_dock(pdb_id, bond_res, work_dir, data_dir)
