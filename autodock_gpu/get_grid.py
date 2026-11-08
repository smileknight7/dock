


from meeko import Polymer
from meeko import MoleculePreparation, ResidueChemTemplates
from meeko import PDBQTWriterLegacy
from pathlib import Path
import subprocess
from utils.file_utils import load_mol, load_idx_list
import argparse
from meeko.gridbox import get_gpf_string, boron_silicon_atompar       # 硼元和硅是补充元素
from utils.data_utils import parse_triplet, compute_box_bbox, remove_protein_H
from .config import AUTOGRID4_BINARY, any_lig_base_types, rec_types
from tqdm import tqdm
import sys
from utils.config import run_command



'''
path 对象的属性

p.name	文件名，带后缀	5REJ_protein_pocket10.pdb
p.stem	文件名，去掉最后一个后缀	5REJ_protein_pocket10
p.suffix	最后一个后缀	.pdb
p.parent	所在目录的完整路径	/data3/.../cov200_test/pocket
p.parent.name	所在目录的名字	pocket


'''







def prepare_receptor(receptor_pdb, output_pdbqt, charge_model=None, allow_bad_res=False):
    """Prepare the rigid receptor PDBQT with Meeko."""
    mk_prep = MoleculePreparation(charge_model=charge_model or "gasteiger")
    chem_templates = ResidueChemTemplates.create_from_defaults()
    pdb_str = remove_protein_H(receptor_pdb)                        # 暂时修补
    mypol = Polymer.from_pdb_string(pdb_str, chem_templates, mk_prep, allow_bad_res=allow_bad_res, default_altloc="A")
    rigid_pdbqt_string, _ = PDBQTWriterLegacy.write_string_from_polymer(mypol)
    if not rigid_pdbqt_string.strip():
        raise ValueError(f"No receptor atoms prepared from {receptor_pdb}")
    Path(output_pdbqt).write_text(rigid_pdbqt_string)




def run_grid(receptor_pdb, output_dir, *,
             reference_lig_file=None, box_center=None, box_size=None,
             padding=5.0, charge_model=None, allow_bad_res=False,
             binary=None):
    receptor_pdb = Path(receptor_pdb).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if reference_lig_file is not None:
        ref_lig_mol = load_mol(reference_lig_file)
        ref_coords = ref_lig_mol.GetConformer().GetPositions()
        box_center, box_size = compute_box_bbox(ref_coords, padding)
    else:
        assert box_center is not None and box_size is not None

    base = receptor_pdb.stem
    output_pdbqt = output_dir / f"{base}.pdbqt"
    output_gpf = output_dir / f"{base}.gpf"

    print(f"Preparing receptor: {receptor_pdb.name}", file=sys.stderr)
    print(f"Box center: ({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f})", file=sys.stderr)
    print(f"Box size:   ({box_size[0]:.3f}, {box_size[1]:.3f}, {box_size[2]:.3f})", file=sys.stderr)
    prepare_receptor(receptor_pdb, output_pdbqt, charge_model, allow_bad_res)

    ff_name = "boron-silicon-atom_par.dat"               # 这里是meeko携带的使用B或者是SI只能用一个进行grid
    (output_dir / ff_name).write_text(boron_silicon_atompar)                            
    used_rec_types = {
        line.split()[-1] for line in output_pdbqt.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    }
    if used_rec_types - set(rec_types):
        raise ValueError(f"Unsupported receptor atom types: {sorted(used_rec_types - set(rec_types))}")
    gpf_string, _ = get_gpf_string(box_center, box_size, output_pdbqt.name,
                                [atom_type for atom_type in rec_types if atom_type in used_rec_types], any_lig_base_types,
                                ff_param_fname=ff_name)
    output_gpf.write_text(gpf_string)

    
    glg_path = output_gpf.with_suffix(".glg")
    print(f"Running autogrid4: {output_gpf.name}", file=sys.stderr)
    run_command([binary or AUTOGRID4_BINARY, "-p", output_gpf, "-l", glg_path],
                output_dir, "autogrid4")

    print(f"OK\t{base}\t{output_pdbqt}")
    print(f"receptor_gpf\t{output_gpf}", file=sys.stderr)
    print(f"grid_log\t{glg_path}", file=sys.stderr)
    return output_pdbqt, output_gpf, glg_path


def main(args):
    has_ref = args.ref_ligand_file is not None or args.ref_ligand_dir is not None
    has_box = args.box_center is not None or args.box_size is not None
    if has_ref == has_box:
        raise ValueError("Use either --reference-ligand or --box-center+--box-size (not both, not neither)")
    if (args.box_center is None) != (args.box_size is None):
        raise ValueError("--box-center and --box-size must be used together")
    if (args.receptor_pdb is None) == (args.receptor_dir is None):
        raise ValueError("Provide --receptor-pdb or --receptor-dir")
    if args.receptor_pdb is not None:
        if args.ref_ligand_dir is not None:
            raise ValueError("Use --ref-ligand-file with --receptor-pdb")
        receptor_pdb = Path(args.receptor_pdb).resolve()
        output_dir = Path(args.output_dir / receptor_pdb.stem).resolve()                # .parent 是父目录的绝对路径  .stem 是文件名不带后缀
        output_dir.mkdir(parents=True, exist_ok=True)
        run_grid(
            receptor_pdb, output_dir,
            reference_lig_file=Path(args.ref_ligand_file).resolve() if args.ref_ligand_file else None,
            box_center=parse_triplet(args.box_center),
            box_size=parse_triplet(args.box_size),
            padding=args.padding,
            charge_model=args.charge_model,
            allow_bad_res=args.allow_bad_res,
            binary=Path(args.binary).resolve() if args.binary else None,
        )

    elif args.receptor_dir is not None:
        if args.ref_ligand_file is not None or args.data_list is None:
            raise ValueError("Use --data_list and optionally --ref-ligand-dir with --receptor-dir")
        failed = False
        for  pdb_id in tqdm(load_idx_list(args.data_list)):
            ref_ligand_file = (args.ref_ligand_dir / pdb_id / f"{pdb_id}_ligand.sdf").resolve() if args.ref_ligand_dir else None
            receptor_pdb = Path(args.receptor_dir / f"{pdb_id}_protein_pocket10.pdb" ).resolve()   # 注意切换后缀名字     !!!!
            output_dir = Path(args.output_dir / receptor_pdb.stem).resolve()
            try:
                run_grid(
                    receptor_pdb, output_dir=output_dir,
                    reference_lig_file=ref_ligand_file,
                    box_center=parse_triplet(args.box_center),
                    box_size=parse_triplet(args.box_size),
                    padding=args.padding,
                    charge_model=args.charge_model,
                    allow_bad_res=args.allow_bad_res,
                    binary=Path(args.binary).resolve() if args.binary else None,
                )
            except Exception as e:
                failed = True
                print(f"Error processing {pdb_id}: {e}", file=sys.stderr)
        return 1 if failed else 0

    else: 
        raise ValueError("Must provide either --receptor-pdb and --ref-ligand-file, or --receptor-dir and --ref-ligand-dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--receptor-pdb",     type=Path, default=None, help="Input receptor PDB (full protein or pocket).")
    parser.add_argument("--ref-ligand-file", type=Path, default=None, help="Reference ligand for box envelope.")
    parser.add_argument("--receptor-dir",     type=Path, default=None, help="Input receptor PDB (full protein or pocket).")
    parser.add_argument("--ref-ligand-dir", type=Path, default=None, help="datadir for reference ligand for box envelope.")
    parser.add_argument("--data_list",        type=Path, default=None, help="Directory containing ligand PDBQT files.")
    
    parser.add_argument("--box-center",       type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--box-size",         type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--padding",          type=float, default=5.0, help="Padding (A) when using --reference-ligand.")
    parser.add_argument("--output-dir",       type=Path, required=True, help="Output directory.")
    parser.add_argument("--allow-bad-res",    action="store_true")
    parser.add_argument("--charge-model",     choices=("gasteiger", "espaloma", "zero", "read"), default=None)
    parser.add_argument("--binary",           type=Path, default=None, help="autogrid4 binary path.")
    raise SystemExit(main(parser.parse_args()))

    # receptor_pdb = /data3/lpw/dock_workdir/autodock_gpu/cov200_test/pocket/5REJ_protein_pocket10.pdb
    # ref-ligand_file = /data3/lpw/data/covdocker/data/processed/bonded/5REJ/5REJ_ligand.sdf
    # ref-ligand_dir = /data3/lpw/data/covdocker/data/processed/bonded
    # receptor_dir = /data3/lpw/dock_workdir/autodock_gpu/cov200_test/pocket




# 调试文件
# python -m autodock_gpu.receptor_prepare --receptor-pdb /data3/lpw/dock_workdir/autodock_gpu/cov200_test/pocket/5REJ_protein_pocket10.pdb --ref-ligand-file /data3/lpw/data/covdocker/data/processed/bonded/5REJ/5REJ_ligand.sdf
