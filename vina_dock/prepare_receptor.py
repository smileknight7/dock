#!/usr/bin/env python3
"""Prepare receptor PDBQT and Vina box config with Meeko.

Examples:
    python vina/prepare_receptor.py \
        --receptor-file /data3/lpw/data/dock_data/test_data/9lpr/9lpr_protein.pdb \
        --reference-ligand /data3/lpw/data/dock_data/test_data/9lpr/9lpr_ligand.sdf \
        --output-dir /data3/lpw/dock_workdir/vina/receptor \
        --output-basename receptor

    python vina/prepare_receptor.py \
        --receptor-file /path/to/protein.pdb \
        --box-center 10 11 12 \
        --box-size 20 20 20 \
        --output-dir /path/to/work/receptor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .vina_utils import (
    compute_box_from_reference,
    ensure_directory,
    parse_triplet,
    prepare_receptor_pdbqt,
    validate_box_args,
    write_vina_box_files,
)


def main(args: argparse.Namespace) -> int:
    receptor_file = args.receptor_file.resolve()
    output_dir = ensure_directory((args.output_dir or receptor_file.parent).resolve())
    basename = args.output_basename or receptor_file.stem
    output_pdbqt = output_dir / f"{basename}.pdbqt"
    box_json = output_dir / f"{basename}.box.json"
    box_txt = output_dir / f"{basename}.vina.txt"

    reference_ligand = args.reference_ligand.resolve() if args.reference_ligand else None
    box_center = parse_triplet(args.box_center, "--box-center")
    box_size = parse_triplet(args.box_size, "--box-size")
    validate_box_args(reference_ligand=reference_ligand, box_center=box_center, box_size=box_size)

    if reference_ligand is not None:
        box_center, box_size = compute_box_from_reference(reference_ligand, args.padding)
        metadata = {"source": "reference_ligand", "reference_ligand": str(reference_ligand), "padding": args.padding}
    else:
        assert box_center is not None and box_size is not None
        metadata = {"source": "manual", "padding": 0.0}

    prepare_receptor_pdbqt(
        receptor_file=receptor_file, output_pdbqt=output_pdbqt,
        charge_model=args.charge_model, allow_bad_res=args.allow_bad_res,
    )
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
    p.add_argument("--receptor-file", type=Path, default='/data3/lpw/dock_workdir/3I03/3I03_protein.pdb', help="Input receptor PDB file.")
    p.add_argument("--reference-ligand", type=Path, default='/data3/lpw/dock_workdir/3I03/3I03_ligand.sdf', help="Reference ligand (.pdb/.sdf/.mol/.mol2/.pdbqt) to derive the docking box.")
    p.add_argument("--box-center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--box-size", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--padding", type=float, default=4.0, help="Box padding (A) around reference ligand.")
    p.add_argument("--output-dir", type=Path, default='/data3/lpw/dock_workdir/vina/receptor')
    p.add_argument("--output-basename", type=str, default=None)
    p.add_argument("--allow-bad-res", action="store_true", help="Delete residues with missing atoms.")
    p.add_argument("--charge-model", choices=("gasteiger", "espaloma", "zero", "read"), default=None)
    try:
        raise SystemExit(main(p.parse_args()))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
