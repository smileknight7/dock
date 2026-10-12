#!/usr/bin/env python3
"""Prepare ligand PDBQT files for AutoDock-GPU with Meeko + RDKit.

Supported inputs:
1. Single ligand file (.sdf/.mol2/.mol/.pdb)
2. Ligand directory batch mode
3. CSV batch mode with a SMILES column
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from utils.file_utils import load_mol, write_pdbqt, load_idx_list, collect_ligand_files, read_smiles_records
from utils.data_utils import prepare_rdkit_molecule
import os



def process_one(ligand_id, source, output_dir, is_smiles):
    output_file = Path(output_dir) / f"{ligand_id}.pdbqt"
    mol = load_mol(source, is_smiles)
    mol = prepare_rdkit_molecule(mol)
    mol.SetProp("_Name", ligand_id)
    write_pdbqt(mol, output_file)
    return str(output_file)




def run_batch(items, output_dir, num_workers, is_smiles):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = {}
    failures = []
    if num_workers == 1:
        for ligand_id, source in tqdm(items, desc="Processing ligands"):
            try:
                success[ligand_id] = process_one(ligand_id, source, output_dir, is_smiles)
            except Exception as exc:
                failures.append((ligand_id, str(exc)))
    else:
        with tqdm_joblib(total=len(items), desc="Processing ligands"):
            results = Parallel(n_jobs=num_workers, return_exceptions=True)(
                delayed(process_one)(ligand_id, source, output_dir, is_smiles)
                for ligand_id, source in items
            )
        for (ligand_id, _), result in zip(items, results):
            if isinstance(result, Exception):
                failures.append((ligand_id, str(result)))
            else:
                success[ligand_id] = result
    return success, failures


def main(args):
    
    if args.ligand_file is not None:
        if args.num_workers != 1:
            raise ValueError("--num_workers is only valid for batch modes")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        path = process_one(Path(args.ligand_file).stem, str(args.ligand_file),
                           args.output_dir, is_smiles=False)
        print(path)
        return 0

    elif args.ligand_dir and args.data_list is not None:                  # 这个是做covdocker临时改的
        items = []
        for id in load_idx_list(args.data_list):
            id_file =os.path.join(args.ligand_dir, id , f'{id}_ligand.sdf')
            if  os.path.exists(id_file):
                items.append((id, id_file))
        is_smiles = False
    elif args.ligand_dir is not None and args.data_list is None:
        items = collect_ligand_files(args.ligand_dir)
        is_smiles = False
    else:
        items = read_smiles_records(args.ligand_csv, args.id_column, args.smiles_column)
        is_smiles = True

    success, failures = run_batch(items, args.output_dir, args.num_workers, is_smiles)
    for ligand_id, path in success.items():
        print(f"OK\t{ligand_id}\t{path}")
    for ligand_id, msg in failures:
        print(f"FAIL\t{ligand_id}\t{msg}", file=sys.stderr)
    print(f"\nDone. OK: {len(success)}, Failed: {len(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-file",   type=Path, help="Single ligand file (.sdf/.mol2/.mol/.pdb).")
    parser.add_argument("--ligand-dir",    type=Path, help="Directory containing ligand files.")
    parser.add_argument("--ligand-csv",    type=Path, help="CSV file containing a SMILES column.")
    parser.add_argument("--output-dir",    type=Path, help="Output directory.")
    parser.add_argument("--smiles-column", default="SMILES", help="CSV column containing SMILES strings.")
    parser.add_argument("--id-column",     default=None, help="Optional CSV column used as ligand IDs.")
    parser.add_argument("--num_workers",   type=int, default=1, help="Parallel workers for batch mode.")
    parser.add_argument("--data_list", help='data_list to docking' )

    raise SystemExit(main(parser.parse_args()))

# 测试方式
# python -m autodock_gpu.ligand_prepare --ligand-dir /data3/lpw/data/covdocker/data/processed/bonded --data_list /home/lpw/ForceFM/data/splits/covdocker/cov200_list.txt --output-dir /data3/lpw/dock_workdir/autodock_gpu/cov200_test/lig_pdbqt 