#!/usr/bin/env python3
"""Prepare receptor PDBQT and Vina box config with Meeko.

Examples:
    python -m vina_dock.prepare_receptor \
        --receptor-file /data3/lpw/data/dock_data/test_data/9lpr/9lpr_protein.pdb \
        --reference-ligand /data3/lpw/data/dock_data/test_data/9lpr/9lpr_ligand.sdf \
        --output-dir /data3/lpw/dock_workdir/vina/receptor \
        --output-basename receptor

    python -m vina_dock.prepare_receptor \
        --receptor-file /path/to/protein.pdb \
        --box-center 10 11 12 \
        --box-size 20 20 20 \
        --output-dir /path/to/work/receptor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from autodock_gpu.get_grid import prepare_receptor
from utils.data_utils import compute_box_bbox, remove_protein_H
from utils.file_utils import load_mol

from .vina_utils import (
    write_vina_box_files,
)


def main(args: argparse.Namespace) -> int:
    receptor_file = args.receptor_file.resolve()
    output_dir = (args.output_dir or receptor_file.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.output_basename or receptor_file.stem
    output_pdbqt = output_dir / f"{basename}.pdbqt"
    box_json = output_dir / f"{basename}.box.json"
    box_txt = output_dir / f"{basename}.vina.txt"

    reference_ligand = args.reference_ligand.resolve() if args.reference_ligand else None
    box_center, box_size = args.box_center, args.box_size
  

    if reference_ligand is not None:
        ref_mol = load_mol(reference_ligand)
        if ref_mol.GetNumConformers() == 0:
            raise ValueError(f"No conformer in {reference_ligand}")
        box_center, box_size = compute_box_bbox(ref_mol.GetConformer().GetPositions(), args.padding)
        metadata = {"source": "reference_ligand", "reference_ligand": str(reference_ligand), "padding": args.padding}
    else:
        assert box_center is not None and box_size is not None
        metadata = {"source": "manual", "padding": 0.0}

    prepare_receptor(receptor_file, output_pdbqt, args.charge_model, args.allow_bad_res)
    write_vina_box_files(
        json_path=box_json, txt_path=box_txt, receptor_pdbqt=output_pdbqt,
        box_center=box_center, box_size=box_size, metadata=metadata,
    )

    print(f"receptor_pdbqt\t{output_pdbqt}")
    print(f"box_json\t{box_json}")
    print(f"box_txt\t{box_txt}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receptor-file", type=Path, required=True, help="Input receptor PDB file.")
    p.add_argument("--reference-ligand", type=Path, help="Reference ligand (.pdb/.sdf/.mol/.mol2) to derive the docking box.")
    p.add_argument("--box-center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--box-size", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--padding", type=float, default=4.0, help="Box padding (A) around reference ligand.")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--output-basename", type=str, default=None)
    p.add_argument("--allow-bad-res", action="store_true", help="Delete residues with missing atoms.")
    p.add_argument("--charge-model", choices=("gasteiger", "espaloma", "zero", "read"), default=None)
    try:
        raise SystemExit(main(p.parse_args()))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
    
    
# python -m vina_dock.prepare_receptor --receptor-file /home/lpw/dock_scripts/autodock/data/3A1D/3A1D_protein.pdb --reference-ligand /home/lpw/dock_scripts/autodock/data/3A1D/3A1D_ligand.sdf --output-dir /data3/lpw/dock_workdir/vina/receptor --allow-bad-res --charge-model gasteiger