import pandas as pd
import numpy as np
from pathlib import Path
from utils.cov_utils import _find_pocket_bond_idx, parse_mol,assign_substruct_coords, generate_conformation
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from traceback import print_exc
import os
import tempfile
from shutil import copyfile, rmtree
from config import AA_ATOMS, run_prepare_covalent, run_prepare_receptor4, run_prepare_flex_receptor4


'''
注释都是按1A0L来构建的



Index(['date', 'src', 'pdb_id', 'pre-reactive smiles', 'bonded protein',
       'bonded pocket', 'bonded chain', 'bonded ligand', 'bond', 'length',
       'products', 'reactants', 'set', 'products_wH', 'products_woAA',
       'products_woAAbackbone'],
      dtype='object')


2ZK2 这个样本第二次preparation的时候报错切割的时候

'''




def compute_and_save_grid_box(out_path, center_coord, ligand_mol,
                              spacing=0.375, margin=5.0, max_npts=126):
      """预计算 autogrid box 并写出 grid_box.txt。

      box 中心取共价原子(如 CYS 的 SG)坐标;半边长按参考配体相对该原子的
      最大伸展 + margin 估计,保证配体绕共价键转到任意方向都落在盒内。

      out_path:     grid_box.txt 输出路径
      center_coord: 共价原子坐标 [x, y, z](即 bond_aa_coord)
      ligand_mol:   带构象的参考配体(用于估计配体尺寸)
      返回:         ((cx, cy, cz), npts)
      """
      cx, cy, cz = (float(c) for c in center_coord)
      lig_xyz = np.asarray(ligand_mol.GetConformer().GetPositions())
      max_reach = np.linalg.norm(lig_xyz - np.asarray(center_coord), axis=1).max()
      npts = int(np.ceil((max_reach + margin) / spacing)) * 2   # autogrid 要求偶数
      npts = min(npts, max_npts)                                # 上限保护,按需调
      with open(out_path, 'w') as fp:
            # 第一行 gridcenter,第二行 npts(立方,xyz 相同)
            fp.write(f'{cx:.3f},{cy:.3f},{cz:.3f}\n{npts},{npts},{npts}')
      return (cx, cy, cz), npts


N_CONFORMERS = 3          # 每个样本保留几个"验证过"的随机构象
MAX_CONF_ATTEMPTS = 20    # 为凑够 N_CONFORMERS,最多尝试多少个随机种子


def _count_lig_atoms_in_flex(flex_pdbqt):
      '''数 flex pdbqt 里的配体重原子数(非 AA 命名且非氢)。
      下游 get_rmsd 用 MolFromPDBBlock(removeHs=True) 去氢后再与 GT 比,
      所以这里也必须排除极性氢(否则像 3BEB 这种带 NH/OH 的配体会被多算),口径才一致。'''
      n = 0
      for l in open(flex_pdbqt):
            if l[:6] not in ('ATOM  ', 'HETATM'):
                  continue
            if l[12:16].strip() in AA_ATOMS:
                  continue
            if l.rstrip().split()[-1].startswith('H'):   # autodock 原子类型(末列)= H/HD,排除氢
                  continue
            n += 1
      return n


def _graft_split_lig_count(conf_mol, pocket_pdb, ligindices, bond_res):
      '''把一个构象 嫁接+拆分(prepare_covalent → receptor4 → flexreceptor4),
      返回 flex 中配体原子数;任何一步失败或没产物返回 -1。全程在临时目录,不留中间文件。'''
      work = Path(tempfile.mkdtemp(prefix='covprep_'))
      try:
            sdf = work / 'cand.sdf'
            Chem.MolToMolFile(conf_mol, str(sdf))
            run_prepare_covalent(['--ligand', str(sdf), '--ligindices', ligindices,
                                  '--receptor', str(pocket_pdb), '--residue', bond_res,
                                  '--outputfile', str(work / 'ligand_covalent.pdb')], cwd=work)
            run_prepare_receptor4(['-r', str(work / 'ligand_covalent.pdb')], cwd=work)
            run_prepare_flex_receptor4(['-r', str(work / 'ligand_covalent.pdbqt'),
                                        '-s', f'ligand_covalent:{bond_res}'], cwd=work)
            flex = work / 'ligand_covalent_flex.pdbqt'
            return _count_lig_atoms_in_flex(flex) if flex.exists() else -1
      except Exception:
            return -1
      finally:
            rmtree(work, ignore_errors=True)


def single_preparation(pdb_id, bond, product_smi, data_dir, output_dir,
                       n_conformers, max_attempts):
      # 处理ligand

      name, resName, chainID, resSeq = bond.split('-')[0].split(' ')
      ligand_file = data_dir / pdb_id / f'{pdb_id}_ligand.sdf'
      ligand_product_smi = product_smi
      ligand_product_mol = Chem.MolFromSmiles(ligand_product_smi) 
      ligand_mol = Chem.SDMolSupplier(str(ligand_file))[0]



      # 处理pocket
      pocket_file = data_dir/ pdb_id /f'{pdb_id}_10Apocket.pdb'
      pocket_obmol, _, pocket_atoms, pocket_coords, pocket_bonds = parse_mol(str(pocket_file), remove_H=True)


      # 处理cov_bond_res ---> pcoket indices（这里主要是为了获取坐标）
      bond_pocket_idx = _find_pocket_bond_idx(pocket_file, [name, resName, chainID, resSeq], ignore_H=True)
      assert bond_pocket_idx != None, f"Cannot find the covalent bond atom in the pocket for {pdb_id} with bond {bond}"                   #  这里要这样做下兜底，否则直接走下面的了
      assert pocket_obmol.GetAtom(bond_pocket_idx+1).GetResidue().GetName() == resName,f'Expect residue {resName} but got {pocket_obmol.GetAtom(bond_pocket_idx+1).GetResidue().GetName()}'
      second_bond_pocket_idx = None
      for b in pocket_bonds:
            if bond_pocket_idx in b[:2]:                                                  # 直接根据res上的反应原子，找与之相连的另一个原子就可以找到第2号原子
                  second_bond_pocket_idx = b[0] if b[1]==bond_pocket_idx else b[1]
                  break
      assert second_bond_pocket_idx != None, 'Cannot find the second atom in the covalent bond in the pocket'


      # 处理ligand    ---> ligand indices(bond_idx ligand端的atom_idx， aa_bond_idx res端的atom_idx)  #  这里就保证都是基于ligand_product_mol的idx了
      lig_idx_list = ligand_product_mol.GetSubstructMatch(ligand_mol)               #  ligand_atom_id (17, 18, 19, 8, 9, 7, 6, 5, 4, 3, 21, 20, 1, 0, 2)
      ligand_product_ids =  [atom.GetIdx() for atom in ligand_product_mol.GetAtoms()]     #  ligand_product_atom_id [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
      lig_idx = set(lig_idx_list)                                            
      bond_idx = set() # the idx to be connected to aa                       
      aa_bond_idx = None                                                                  # 这里用的思想其实和openfold的思想是一致的
      for atom in lig_idx:
            atom_obj = ligand_product_mol.GetAtomWithIdx(atom)
            #check whether the atom is the connect atom between aa and ligand
            bonds = atom_obj.GetBonds()
            for bond in bonds:                                                           # 这里要注意是按bond取的
                  b = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])                # 包装成set{12, 30}和ligand部分的atom_idx比较
                  connect_idx = b - (b & lig_idx)                                        # (b & lig_idx) ligand端原子
                  if connect_idx != set():                                               # 这里就能得到ligand端的bond_idx
                        bond_idx = b - connect_idx
                        aa_bond_idx = connect_idx
                        break
                  # print(connect_idx )
      assert aa_bond_idx != None, 'Cannot find the bond idx in ligand that connects to aa'

      # second_bond_idx 找连接bond_idx的邻居
      second_bond_idx = set()
      for bond in ligand_product_mol.GetAtomWithIdx(next(iter(bond_idx))).GetBonds():
            second_bond_idx = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]) & lig_idx - bond_idx  
            if second_bond_idx != set():
                  break
      assert second_bond_idx != set(), 'Cannot find the second connect atom for ligand'

      # second_aa_bond_idx 找连接aa_bond_idx的邻居
      second_aa_bond_idx = set()
      for bond in ligand_product_mol.GetAtomWithIdx(next(iter(aa_bond_idx))).GetBonds():
            second_aa_bond_idx = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]) - bond_idx - aa_bond_idx
            if second_aa_bond_idx != set():
                  break
      assert second_aa_bond_idx != set(), 'Cannot find the second connect atom for aa'

      print('ligand_bond_idx', bond_idx)
      print('aa_bond_idx', aa_bond_idx)
      print('second_ligand_bond_idx', second_bond_idx)

      # assign coords
      bond_aa_coord = pocket_coords[bond_pocket_idx]                      # 这里是得到了pocket中共价原子的坐标的
      second_bond_aa_coord = pocket_coords[second_bond_pocket_idx]       
      assign_substruct_coords(ligand_product_mol, list(lig_idx_list) + list(aa_bond_idx) + list(second_aa_bond_idx), ligand_mol.GetConformer().GetPositions().tolist() + [bond_aa_coord] + [second_bond_aa_coord])


      # Convert sets to single values first                             拆包（拆成单个int）
      bond_idx = next(iter(bond_idx))                                   # 8
      aa_bond_idx = next(iter(aa_bond_idx))                             # 10
      second_aa_bond_idx = next(iter(second_aa_bond_idx))               # 7
      second_bond_idx = next(iter(second_bond_idx))                     # 11


      # Build old_to_new index mapping BEFORE removing atoms
      # This is critical because after RemoveAtom, the indices in autodock_lig_mol will be 0,1,2,3,...
      # We need to map from original indices in ligcovalent_mol to new indices in autodock_lig_mol
      kept_atoms_list = sorted(lig_idx | {aa_bond_idx} | {second_aa_bond_idx})                              # 重新排序一下[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21]
      old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_atoms_list)}       # new部分是裁剪之后的ligand索引{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16}


      # get the ligand part that used in autodock (ligand part plus one bonded aa atom)
      edit_mol = Chem.EditableMol(copy.deepcopy(ligand_product_mol))
      atoms_to_remove = set(range(ligand_product_mol.GetNumAtoms())) - lig_idx - {aa_bond_idx} - {second_aa_bond_idx}
      edit_mol.BeginBatchEdit()
      for atom in sorted(atoms_to_remove, reverse=True):  # Remove from back to front to avoid index shifting
            edit_mol.RemoveAtom(atom)
      edit_mol.CommitBatchEdit()
      autodock_lig_mol = edit_mol.GetMol()                # 删掉原子之后会重排序(重排顺序就是像再pdb中删除atom这种)
      #atom_idx = [atom.GetIdx() for atom in autodock_lig_mol.GetAtoms()]


      # Use correct mapping to calculate new indices in autodock_lig_mol (1-based for prepareCovalent.py)
      # bond_idx is in ligcovalent_mol, we need to find its position in autodock_lig_mol
      autodock_bond_idx = old_to_new_idx[bond_idx] + 1  # ligand's bonding atom
      autodock_second_bond_idx = old_to_new_idx[second_bond_idx] + 1  # ligand's second atom
      autodock_aa_idx = old_to_new_idx[aa_bond_idx] + 1  # AA bonding atom (e.g., SG for CYS)
      autodock_second_aa_idx = old_to_new_idx[second_aa_bond_idx] + 1  # AA second atom (e.g., CB for CYS)

      output_i_dir = output_dir / pdb_id
      os.makedirs(output_i_dir, exist_ok=True)

      try:
            # 预计算 autogrid box(中心=共价原子 SG),供后续对接直接加载
            compute_and_save_grid_box(output_i_dir / 'grid_box.txt', bond_aa_coord, ligand_mol)

            # ligindices should point to AA atoms in the SDF file for alignment with receptor
            # Format: X,Y where X-Y are the two atoms to align (Y is the bonding atom to receptor)
            # Since autodock_lig_mol contains AA atoms (CB-SG for CYS), we use AA indices
            with open(output_i_dir / 'ligand_indices.txt', 'w') as fp:
                  fp.write(f'{autodock_second_aa_idx},{autodock_aa_idx}')

            # autodock_lig_mol = Chem.AddHs(autodock_lig_mol)
            Chem.SanitizeMol(autodock_lig_mol)
            Chem.MolToMolFile(autodock_lig_mol, str(output_i_dir / 'autodock_gt.sdf'))

            copyfile(pocket_file, output_i_dir / 'pocket.pdb')   # 先放好 pocket,验证嫁接时要复用

            # 反复用不同随机种子嵌入构象,只保留"嫁接+拆分后配体原子不丢"的,凑够 n_conformers 个
            bond_res = f'{chainID}:{resName}{resSeq}'
            ligindices = f'{autodock_second_aa_idx},{autodock_aa_idx}'
            expected_n = ligand_mol.GetNumAtoms()                # 配体重原子数(= flex 里应有的配体原子数)
            n_valid = 0
            for seed in range(max_attempts):
                  cand = generate_conformation(autodock_lig_mol, seed=seed)   # 不改原 mol,每次返回新构象
                  if _graft_split_lig_count(cand, output_i_dir / 'pocket.pdb', ligindices, bond_res) == expected_n:
                        n_valid += 1
                        Chem.MolToMolFile(cand, str(output_i_dir / f'autodock_randomConform_{n_valid}.sdf'))
                        if n_valid == n_conformers:
                              break
            if n_valid < n_conformers:
                  raise RuntimeError(f'{pdb_id}: 仅找到 {n_valid}/{n_conformers} 个有效构象(拆分不丢原子),已试 {max_attempts} 个种子')


            

            
      except Exception:
            rmtree(output_i_dir, ignore_errors=True)   # 失败回滚:删掉残缺目录,保证"目录存在 == 产物完整"
            raise                                           # 继续抛出,让 batch 层 _run_one 如实标记为失败

if __name__ == '__main__':
      info_file = Path('/home/lpw/dock_scripts/autodock/autodock_cov/data/covdocker_info_common.csv')   # 1A0L不在交叉的数据集中，需要从dataset.filtered.random_split.csv 这个表中加载
      data_dir = Path('/data3/lpw/data/covdocker/data/processed/bonded')
      output_dir = Path('/data3/lpw/dock_workdir/autodock_cov/preparation5')
      df_info = pd.read_csv(info_file)
      
      pdb_id = '2ZK2'
      n_conformers = 3
      max_attempts = 20
      bond = df_info.loc[df_info['pdb_id'] == pdb_id, 'bond'].values[0]                               # OG SER D 195-C2 APA D 301'
      name, resName, chainID, resSeq = bond.split('-')[0].split(' ')    
      product_smi = df_info.loc[df_info['pdb_id'] == pdb_id, 'products'].values[0]

      single_preparation(pdb_id, bond, product_smi, data_dir, output_dir, n_conformers=n_conformers, max_attempts=max_attempts)
