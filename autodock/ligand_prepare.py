#!/usr/bin/env python3
"""Prepare ligand PDBQT files for AutoDock4 with RDKit + MGLTools.

Supported inputs:
1. Single ligand file
2. Ligand directory batch mode                  使用这个mode的时候支持sdf，mol2，mol，pdb格式的文件
3. CSV batch mode with a SMILES column
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from utils.file_utils import load_mol
from ..utils.config import PREPARE_LIGAND4, run_mgltools_script
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
from meeko import MoleculePreparation, PDBQTWriterLegacy
from utils.data_utils import generate_conformer, prepare_ligand




# meeko做法
# def single_ligand_run(cache_file, output_file, cache_dir):
#     run_mgltools_script(PREPARE_LIGAND4, ["-l", str(cache_file), "-o", str(output_file)], cache_dir)


def prepare_ligand_file(ligand_file, cache_dir, output_dir):
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ligand_id = Path(ligand_file).stem
    cache_file = cache_dir / f"{ligand_id}.pdb"
    output_file = output_dir / f"{ligand_id}.pdbqt"
    ligand_mol = load_mol(ligand_file)
    ligand_mol = generate_conformer(ligand_mol, str(ligand_file))
    Chem.MolToPDBFile(ligand_mol, str(cache_file))
    prepare_ligand(ligand_mol, output_file)
    # autodock的处理方法
    # single_ligand_run(cache_file, output_file, cache_dir)
    return 0



def prepare_ligand_files(ligand_dir):
    ligand_dir = Path(ligand_dir)
    success_list = []
    for ligand_file in ligand_dir.iterdir():            # 小分子最好使用sdf/mol2格式，使用pdb格式可能会出错
        lig_id = ligand_file.stem
        try:
            ligand_mol = load_mol(str(ligand_file))
            ligand_mol = generate_conformer(ligand_mol, str(ligand_file))
            success_list.append((lig_id, ligand_mol))
        except Exception as e:
            print(f"FAIL {lig_id}: {e}", file=sys.stderr)
    return success_list




def read_smiles_records(csv_path, id_column=None, smiles_column="smiles"):

    df = pd.read_csv(csv_path)
    items = [(str(row[id_column]) if id_column else f'ligand_{i:04d}',
              str(row[smiles_column])) for i, row in df.iterrows()]
    success_list = []
    for item in items:
        lig_id = item[0]
        try:
            lig_mol = load_mol(item, is_smiles=True)
            success_list.append(lig_id, lig_mol)
        except Exception as e:
            print(f"FAIL {item[0]}: {e}", file=sys.stderr)
    return success_list





def run_batch(ligand_list, cache_dir, output_dir, num_workers):
    '''Run batch ligand preparation in parallel. Returns (success_dict, failure_list).'''
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = {}
    failures = []
    if num_workers == 1:
        for lig_id, lig_mol in tqdm(ligand_list, desc="Processing ligands"):
            prepared_file = output_dir / f"{lig_id}.pdbqt"
            try:
                prepare_ligand( lig_mol, prepared_file)
                success[lig_id] = str(prepared_file)
            except Exception as exc:
                failures.append((lig_id, str(exc)))
    else:
        with tqdm_joblib(total=len(ligand_list), desc="Processing ligands"):              # ligand_list 这个就相当于是jobs好吧
            results = Parallel(n_jobs=num_workers, return_exceptions=True)(
                delayed(prepare_ligand)(
                    lig_mol, 
                    output_dir / f"{lig_id}.pdbqt",
                )
                for lig_id, lig_mol in ligand_list
            )
        for lig, result in zip(ligand_list, results):
            prepared_file = output_dir / f"{lig[0]}.pdbqt"
            if isinstance(result, Exception):
                failures.append((lig[0], str(result)))
            else:
                success[lig[0]] = str(prepared_file)

    return success, failures


def main(args: argparse.Namespace) -> int:
    if args.ligand_file is not None:
        if args.num_workers != 1:
            raise ValueError("--jobs is only valid for batch modes")
        return prepare_ligand_file(args.ligand_file, args.cache_dir, args.output_dir)

    if args.output_dir is None:
        raise ValueError("--output-dir is required for --ligand-dir and --csv")

    if args.ligand_dir is not None:
        inputs = prepare_ligand_files(args.ligand_dir, args.cache_dir)
    else:
        inputs = read_smiles_records(args.ligand_csv, args.cache_dir, args.id_column, args.smiles_column)

    success, failures = run_batch(inputs, args.cache_dir, args.output_dir, args.num_workers)
    for ligand_id, path in success.items():
        print(f"OK\t{ligand_id}\t{path}")
    for ligand_id, msg in failures:
        print(f"FAIL\t{ligand_id}\t{msg}", file=sys.stderr)
    print(f"\nDone. OK: {len(success)}, Failed: {len(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # source_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--ligand-file", type=str, help="Single ligand file (.sdf/.mol2/.mol/.pdb).")
    parser.add_argument("--ligand-dir", type=Path, help="Directory containing ligand files.")
    parser.add_argument("--ligand-csv", type=Path, default='/data3/lpw/data/dock_data/zinc_50.csv', help="CSV file containing a SMILES column.")
    parser.add_argument("--cache-dir", type=Path, default='/data3/lpw/dock_workdir/autodock/cache', help="Directory for intermediate files.")
    parser.add_argument("--output-dir", type=Path, default='/data3/lpw/dock_workdir/autodock/test/ligands', help="Output directory for batch mode or single-file mode.")
    parser.add_argument("--smiles-column", default="SMILES", help="CSV column containing SMILES strings.")
    parser.add_argument("--id-column", default=None, help="Optional CSV column used as ligand IDs.")
    parser.add_argument("--num_workers", type=int, default=10, help="Parallel workers for batch mode.")
    raise SystemExit(main(parser.parse_args()))
