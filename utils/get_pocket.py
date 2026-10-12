#!/usr/bin/env python3
"""Crop a binding pocket PDB. Unified across autodock / autodock_gpu / vina.

Cuts protein residues whose CA falls within --pocket-cutoff Angstrom of any
ligand atom, then writes the pocket as a PDB. Box info is printed to stdout
(informational only, not consumed by any downstream script).

Examples:
    python -m utils.get_pocket \\
        --ligand-file /path/to/ref.sdf \\
        --protein-file /path/to/protein.pdb \\
        --output-dir /path/to/pockets \\
        --pocket-cutoff 10.0
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from tqdm import tqdm
from utils.data_utils import parse_protein, select_pocket, compute_box_bbox
from utils.file_utils import load_mol, write_pocket_pdb, load_idx_list

def crop_pocket(ligand_file, protein_file, output_dir, pocket_cutoff=10.0,
                box_padding=4.0, output_basename=None):
    """Crop pocket PDB and return (pocket_pdb_path, box_center, box_size)."""
    ligand_file = Path(ligand_file).resolve()                                      # 这个就是将路径解析成绝对路径
    protein_file = Path(protein_file).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lig_mol = load_mol(ligand_file)
    lig_coords = lig_mol.GetConformer().GetPositions()
    protein_atoms = parse_protein(protein_file)
    selected = select_pocket(protein_atoms, lig_coords, pocket_cutoff)
    basename = output_basename or f"{protein_file.stem}_pocket{int(pocket_cutoff)}"
    pocket_pdb = output_dir / f"{basename}.pdb"
    write_pocket_pdb(selected, pocket_pdb)

    box_center, box_size = compute_box_bbox(lig_coords, box_padding)
    return pocket_pdb, box_center, box_size


def main(args):


    if args.ligand_file and args.protein_file is not None:
        pocket_pdb, center, size = crop_pocket(
            args.ligand_file, args.protein_file, args.output_dir,
            pocket_cutoff=args.pocket_cutoff,
            box_padding=args.box_padding,
            output_basename=args.output_basename,
        )
        print(f"pocket_pdb\t{pocket_pdb}")
        print(f"box_center\t{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}")
        print(f"box_size\t{size[0]:.3f},{size[1]:.3f},{size[2]:.3f}")
    elif args.data_dir is not None:
        data_list = load_idx_list(args.data_list)
        for idx in tqdm(data_list):                                         #  这里有空个可以给它配一个多进程？？？？
            ligand_file = args.data_dir / f"{idx}/{idx}_ligand.sdf"
            protein_file = args.data_dir / f"{idx}/{idx}_protein.pdb"
            try:
                pocket_pdb, center, size = crop_pocket(
                    ligand_file, protein_file, args.output_dir,
                    pocket_cutoff=args.pocket_cutoff,
                    box_padding=args.box_padding,
                    output_basename=args.output_basename,
                )
            except Exception as e:
                print(f"Error processing {idx}: {e}", file=sys.stderr)
                continue




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ligand-file",     type=Path, help="Reference ligand (.sdf/.mol2/.mol/.pdb/.pdbqt).")
    parser.add_argument("--protein-file",    type=Path, help="Protein PDB file.")
    parser.add_argument("--data-dir",   type=Path, help="covdokcer dataset")
    parser.add_argument("--data-list",  type=Path, help="data list")
    parser.add_argument("--output-dir",      type=Path, required=True, help="Output directory for pocket PDB.")
    parser.add_argument("--output-basename", default=None, help="Basename for pocket PDB. Default: <protein_stem>_pocket<cutoff>.")
    parser.add_argument("--pocket-cutoff",   type=float, default=10.0, help="CA-distance cutoff in Angstrom.")
    parser.add_argument("--box-padding",     type=float, default=4.0,  help="Padding around ligand bounding box (informational).")
    main(parser.parse_args())

# python -m utils.get_pocket --ligand-file /data3/lpw/data/covdocker/data/processed/bonded/5REJ/5REJ_ligand.sdf --protein-file /data3/lpw/data/covdocker/data/processed/bonded/5REJ/5REJ_protein.pdb --output-dir /data3/lpw/dock_workdir/autodock_gpu/cov200_test/pocket

# python -m utils.get_pocket --data-dir /data3/lpw/data/covdocker/data/processed/bonded --data-list /home/lpw/ForceFM/data/splits/covdocker/cov200_list.txt --output-dir /data3/lpw/dock_workdir/autodock_gpu/cov200_test/pocket