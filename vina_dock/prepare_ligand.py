#!/usr/bin/env python3
"""Prepare ligand PDBQT files for AutoDock Vina with Meeko + RDKit.

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
from meeko import MoleculePreparation, PDBQTWriterLegacy
from utils.data_utils import prepare_ligand, generate_conformer
from utils.file_utils import load_mol







def process_one(ligand_id, source, output_dir, is_smiles):
    output_file = Path(output_dir) / f"{ligand_id}.pdbqt"
    mol = load_mol(source, is_smiles)
    mol = generate_conformer(mol)
    mol.SetProp("_Name", ligand_id)
    prepare_ligand(mol, output_file)
    return str(output_file)


def collect_ligand_files(ligand_dir):
    return [(f.stem, str(f)) for f in Path(ligand_dir).iterdir()
            if f.suffix.lower() in {".sdf", ".mol2", ".mol", ".pdb"}]


def read_smiles_records(csv_path, id_column=None, smiles_column="SMILES"):
    df = pd.read_csv(csv_path)
    items = [(str(row[id_column]) if id_column else f"ligand_{i:04d}",
              str(row[smiles_column])) for i, row in df.iterrows()]
    return [(ligand_id, smi) for ligand_id, smi in items if smi.strip()]


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
    if not any((args.ligand_file, args.ligand_dir, args.ligand_csv)):
        raise ValueError("Provide --ligand-file, --ligand-dir, or --ligand-csv")
    if args.ligand_file is not None:
        if args.num_workers != 1:
            raise ValueError("--num_workers is only valid for batch modes")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        path = process_one(Path(args.ligand_file).stem, str(args.ligand_file),
                           args.output_dir, is_smiles=False)
        print(path)
        return 0
    
    if args.ligand_dir is not None:
        items = collect_ligand_files(args.ligand_dir)
        if args.limit is not None:
            items = items[:args.limit]
        is_smiles = False
    else:
        items = read_smiles_records(args.ligand_csv, args.id_column, args.smiles_column)
        if args.limit is not None:
            items = items[:args.limit]
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
    parser.add_argument("--limit",         type=int, help= 'limit of ligand')
    parser.add_argument("--output-dir",    type=Path, required=True, help="Output directory.")
    parser.add_argument("--smiles-column", default="SMILES", help="CSV column containing SMILES strings.")
    parser.add_argument("--id-column",     default=None, help="Optional CSV column used as ligand IDs.")
    parser.add_argument("--num_workers",   type=int, default=1, help="Parallel workers for batch mode.")
    raise SystemExit(main(parser.parse_args()))

# 示例

# python -m vina_dock.prepare_ligand --ligand-csv /home/lpw/dock_scripts/autodock/data/zinc.csv --limit 10 --output-dir /data3/lpw/dock_workdir/vina/ligand
