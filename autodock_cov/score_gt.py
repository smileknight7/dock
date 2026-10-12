from pathlib import Path
from config import run_prepare_receptor4, run_prepare_flex_receptor4, run_prepare_dpf4, run_prepare_gpf4,run_autogrid4, run_autodock4, run_pdbq2pdb
from utils.cov_utils import parse_epdb_energy, replace_file_content
from rdkit import Chem
import pandas as pd
import os
from tqdm import tqdm
'''
应该还要从对接结果中加上采样构象最丰富的5个pose
covalent.py脚本是有点问题的，需要对脚本进行一点修改
但是当前这个好像只能测试redock


单点计算一次能量值需要花费 40min左右
'''


def _convert_dpf_to_epdb(dpf_file):
    """Use AutoDock4 epdb mode to score the current covalent-flex pose."""
    dpf_file = Path(dpf_file)
    new_lines = []
    has_epdb = False
    for line in dpf_file.read_text().splitlines():
        stripped = line.strip()
        key = stripped.split(maxsplit=1)[0] if stripped else ''
        if key == 'ga_run':
            continue
        if key == 'analysis':
            new_lines.append('epdb')
            has_epdb = True
        else:
            new_lines.append(line)
    if not has_epdb:
        new_lines.append('epdb')
    dpf_file.write_text('\n'.join(new_lines) + '\n')


def _save_scored_pose_from_dlg(dlg_file, pose_i_dir):
    """Extract the epdb-scored flex residue structure from AutoDock's DLG."""
    pose_i_dir = Path(pose_i_dir)
    pose_i_dir.mkdir(parents=True, exist_ok=True)
    pose_pdbqt = pose_i_dir / 'gt_scored_pose.pdbqt'

    prefix = 'INPUT-FLEXRES-PDBQT:'
    pose_lines = []
    for line in Path(dlg_file).read_text().splitlines():
        if line.startswith(prefix):
            pose_lines.append(line[len(prefix):].lstrip())

    if not pose_lines:
        raise ValueError(f'No INPUT-FLEXRES-PDBQT block found in {dlg_file}')

    pose_pdbqt.write_text('\n'.join(pose_lines) + '\n')
    run_pdbq2pdb(['-f', pose_pdbqt.name], cwd=pose_i_dir)
    return pose_pdbqt, pose_pdbqt.with_suffix('.pdb')


def _split_bond_res(bond_res):
    chain_id, residue = bond_res.split(':', 1)
    res_name = residue[:3]
    res_seq = residue[3:]
    return chain_id, res_name, res_seq


def _pdb_element(atom_name, line):
    element = line[76:78].strip()
    if element:
        return element
    return ''.join(c for c in atom_name if c.isalpha())[:1].upper()


def _pdb_atom_record(serial, atom_name, res_name, chain_id, res_seq, x, y, z, element):
    return (
        f'HETATM{serial:5d} {atom_name:>4} {res_name:>3} {chain_id:1}{res_seq:>4}'
        f'    {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}  '
    )


def _read_pocket_residue_atoms(pocket_file, bond_res):
    chain_id, res_name, res_seq = _split_bond_res(bond_res)
    atoms = []
    for line in Path(pocket_file).read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        atom_name = line[12:16].strip()
        if (
            line[17:20].strip() != res_name
            or line[21].strip() != chain_id
            or line[22:26].strip() != res_seq
            or atom_name.startswith(('H', 'D'))
        ):
            continue
        atoms.append({
            'name': atom_name,
            'x': float(line[30:38]),
            'y': float(line[38:46]),
            'z': float(line[46:54]),
            'element': _pdb_element(atom_name, line),
        })
    if not atoms:
        raise ValueError(f'Failed to find residue {bond_res} in {pocket_file}')
    return atoms


def _nearest_residue_atom_name(coord, residue_atoms, excluded_names=()):
    excluded_names = set(excluded_names)
    best_atom = None
    best_dist2 = float('inf')
    for atom in residue_atoms:
        if atom['name'] in excluded_names:
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
        raise ValueError('Failed to match SDF anchor atom to pocket residue atom')
    return best_atom['name']


def _residue_bond_cutoff(element_a, element_b):
    if 'S' in {element_a.upper(), element_b.upper()}:
        return 2.2
    return 1.95


def _write_gt_covalent_pdb(sdf_file, pocket_file, pdb_file, bond_res, ligand_indices, bond_atom_name):
    """Write aligned GT SDF plus complete covalent residue for prepare_flexreceptor4."""
    mol = Chem.SDMolSupplier(str(sdf_file), removeHs=False)[0]
    if mol is None:
        raise ValueError(f'Failed to read GT ligand SDF: {sdf_file}')

    chain_id, res_name, res_seq = _split_bond_res(bond_res)
    residue_atoms = _read_pocket_residue_atoms(pocket_file, bond_res)
    second_aa_idx, aa_idx = [int(i) - 1 for i in ligand_indices.split(',')]
    conf = mol.GetConformer()

    aa_coord = conf.GetAtomPosition(aa_idx)
    if bond_atom_name is None:
        bond_atom_name = _nearest_residue_atom_name(
            (aa_coord.x, aa_coord.y, aa_coord.z),
            residue_atoms,
        )

    second_coord = conf.GetAtomPosition(second_aa_idx)
    second_atom_name = _nearest_residue_atom_name(
        (second_coord.x, second_coord.y, second_coord.z),
        residue_atoms,
        excluded_names=(bond_atom_name,),
    )

    atom_lines = []
    residue_serials = {}
    serial = 1
    element_counts = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        element = atom.GetSymbol()
        if idx == aa_idx:
            atom_name = bond_atom_name
        elif idx == second_aa_idx:
            atom_name = second_atom_name
        else:
            element_counts[element] = element_counts.get(element, 0) + 1
            atom_name = f'{element}{element_counts[element]}'
            atom_name = atom_name[:4]

        pos = conf.GetAtomPosition(idx)
        atom_lines.append(_pdb_atom_record(
            serial, atom_name, res_name, chain_id, res_seq,
            pos.x, pos.y, pos.z, element,
        ))
        if atom_name in {bond_atom_name, second_atom_name}:
            residue_serials[atom_name] = {
                'serial': serial,
                'x': pos.x,
                'y': pos.y,
                'z': pos.z,
                'element': element,
            }
        serial += 1

    for atom in residue_atoms:
        if atom['name'] in residue_serials:
            continue
        atom_lines.append(_pdb_atom_record(
            serial, atom['name'], res_name, chain_id, res_seq,
            atom['x'], atom['y'], atom['z'], atom['element'],
        ))
        residue_serials[atom['name']] = {'serial': serial, **atom}
        serial += 1

    bond_pairs = set()
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx() + 1
        b = bond.GetEndAtomIdx() + 1
        bond_pairs.add(tuple(sorted((a, b))))

    residue_serial_items = list(residue_serials.values())
    for i, atom_a in enumerate(residue_serial_items):
        for atom_b in residue_serial_items[i + 1:]:
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

    Path(pdb_file).write_text('\n'.join(atom_lines + conect_lines + ['END']) + '\n')


def single_dock(pdb_id, bond_res, work_dir,data_dir, pose_dir=None):
    data_i_dir = Path(data_dir/pdb_id)
    work_i_dir = Path(work_dir/pdb_id)
    pose_i_dir = Path(pose_dir or Path(work_dir).parent/'pose')/pdb_id
    os.makedirs(work_i_dir, exist_ok=True)
    with open(data_i_dir/'ligand_indices.txt', 'r') as f:
        ligand_indices = f.read().strip()
    # 预计算好的 grid box(中心=共价原子 SG;第一行 gridcenter,第二行 npts)
    grid_center, grid_npts = (data_i_dir/'grid_box.txt').read_text().splitlines()


    # autodock_gt.sdf is already aligned by single_preparation.py, so do not run
    # prepareCovalent.py again. Running it here can create NaN/zero coordinates.
    _write_gt_covalent_pdb(
        data_i_dir/'autodock_gt.sdf',
        data_i_dir/'pocket.pdb',
        work_i_dir/'ligand_covalent_gt.pdb',
        bond_res,
        ligand_indices,
        bond_atom_name=None,
    )

    run_prepare_receptor4(['-r', f'{data_i_dir}/pocket.pdb', '-o', f'{work_i_dir}/pocket.pdbqt'], cwd=Path(work_i_dir))
    run_prepare_receptor4(['-r', f'{work_i_dir}/ligand_covalent_gt.pdb'], cwd=Path(work_i_dir))


    # 第三步 构建flex部分
    run_prepare_flex_receptor4(['-r', f'{work_i_dir}/pocket.pdbqt', '-s', f'pocket:{bond_res}'], cwd=Path(work_i_dir))
    run_prepare_flex_receptor4(['-r', f'{work_i_dir}/ligand_covalent_gt.pdbqt', '-s', f'ligand_covalent_gt:{bond_res}'], cwd=Path(work_i_dir))





    # 第四步构建dpf，和gpf文件
    run_prepare_dpf4(['-r', 'pocket_rigid.pdbqt', 
                        '-x', 'ligand_covalent_gt_flex.pdbqt', 
                        '-l',  'ligand_covalent_gt_flex.pdbqt', 
                        '-o', 'autodock_gt_score.dpf',
                        '-p', 'move=empty',
                        '-p', 'ga_run=1'], cwd=Path(work_i_dir))
    run_prepare_gpf4(['-r', 'pocket_rigid.pdbqt',
                        '-l',  'ligand_covalent_gt_flex.pdbqt',
                        '-p', f'gridcenter={grid_center}',
                        '-p', f'npts={grid_npts}',
                        '-o', 'autogrid_gt_score.gpf'], cwd=Path(work_i_dir))

    # # 第五步进行对接

    (work_i_dir / 'empty').touch()          #  dock the covalent ligand as a flexible residue and ignore any 'true' ligand ("move='empty'")
    # manually edited to set the appropriate energy model so that the docking score corresponds to the interaction between the flexible residue (the ligand) and the rigid receptor
    replace_file_content(work_i_dir/'autogrid_gt_score.gpf', 'unbound_model bound', 'unbound_energy 0.0')
    replace_file_content(work_i_dir/'autodock_gt_score.dpf', 'unbound_model extended', 'unbound_energy 0.0')
    _convert_dpf_to_epdb(work_i_dir/'autodock_gt_score.dpf')

    run_autogrid4(work_i_dir/'autogrid_gt_score.gpf', work_i_dir/'autogrid_gt_score.glg')
    run_autodock4(work_i_dir/'autodock_gt_score.dpf', work_i_dir/'autodock_gt_score.dlg')
    _save_scored_pose_from_dlg(work_i_dir/'autodock_gt_score.dlg', pose_i_dir)
    return parse_epdb_energy(work_i_dir/'autodock_gt_score.dlg')


def get_first_pose_metrics(pdb_id, pose_dir):
    rmsd_file = Path(pose_dir)/pdb_id/f'{pdb_id}_docking_rmsd.csv'
    if not rmsd_file.exists():
        return None, None
    df_rmsd = pd.read_csv(rmsd_file)
    if df_rmsd.empty:
        return None, None
    first_energy = float(df_rmsd.iloc[0]['energy']) if 'energy' in df_rmsd.columns else None
    first_rmsd = float(df_rmsd.iloc[0]['rmsd']) if 'rmsd' in df_rmsd.columns else None
    return first_energy, first_rmsd


def get_bond_res(df_info, pdb_id):
    rows = df_info[df_info['pdb_id'] == pdb_id]
    if rows.empty:
        raise ValueError(f'{pdb_id} not found in info file')
    bond_info = rows['bond'].values[0]
    name, resName, chainID, resSeq = bond_info.split('-')[0].split(' ')
    return f'{chainID}:{resName}{resSeq}'











if __name__ == "__main__":

    data_dir = Path('/data3/lpw/dock_workdir/autodock_cov/preparation4')
    work_dir = Path('/data3/lpw/dock_workdir/autodock_cov/result4')
    pose_dir = Path('/data3/lpw/dock_workdir/autodock_cov/pose4')
    info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')
    os.makedirs(work_dir, exist_ok=True)
    df_info = pd.read_csv(info_file)
    rows = []
    print('pdb_id,score,first_energy,first_rmsd,status')

    pdb_ids = []
    for pdb_id in pose_dir.iterdir():
        if pdb_id.is_dir():
            pdb_ids.append(pdb_id.name)

    for pdb_id in tqdm(pdb_ids, desc='scoring', unit='pdb'):
        try:
            bond_res = get_bond_res(df_info, pdb_id)
            score = single_dock(pdb_id, bond_res, work_dir, data_dir, pose_dir)
            first_energy, first_rmsd = get_first_pose_metrics(pdb_id, pose_dir)
            rows.append({
                'pdb_id': pdb_id,
                'score': score,
                'first_energy': first_energy,
                'first_rmsd': first_rmsd,
                'status': 'ok',
            })
            print(f'{pdb_id},{score},{first_energy},{first_rmsd},ok', flush=True)
        except Exception as exc:
            rows.append({
                'pdb_id': pdb_id,
                'score': None,
                'first_energy': None,
                'first_rmsd': None,
                'status': str(exc).replace('\n', ' | '),
            })
            print(f'{pdb_id}failed: {exc}', flush=True)

    pd.DataFrame(rows).to_csv(pose_dir/'score_gt_summary5.csv', index=False)
