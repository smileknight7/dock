


from meeko import Polymer
from meeko import MoleculePreparation, ResidueChemTemplates
from meeko import PDBQTWriterLegacy
from pathlib import Path
import math
import subprocess
from utils.file_utils import load_mol, load_idx_list
import argparse
from utils.data_utils import parse_triplet, compute_box_bbox
from .config import AUTOGRID4_BINARY, any_lig_base_types
from tqdm import tqdm
import sys




'''
path 对象的属性

p.name	文件名，带后缀	5REJ_protein_pocket10.pdb
p.stem	文件名，去掉最后一个后缀	5REJ_protein_pocket10
p.suffix	最后一个后缀	.pdb
p.parent	所在目录的完整路径	/data3/.../cov200_test/pocket
p.parent.name	所在目录的名字	pocket


'''


from utils.config import GRID_SPACING, run_command, run_mgltools_script


PREPARE_GPF4ZN = Path(__file__).with_name("prepare_gpf4zn.py")
ZINC_PSEUDO = Path(__file__).with_name("zinc_pseudo.py")
AD4ZN_PARAMETERS = Path(__file__).with_name("AD4Zn.dat")






def prepare_receptor(receptor_pdb, output_pdbqt, charge_model=None, allow_bad_res=False):
    """Prepare the rigid receptor PDBQT with Meeko."""
    output_pdbqt = Path(output_pdbqt)
    output_dir = output_pdbqt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    mk_prep = MoleculePreparation(charge_model=charge_model or "gasteiger")
    chem_templates = ResidueChemTemplates.create_from_defaults()
    pdb_str = Path(receptor_pdb).read_text()                                    # 这里的输入要求是pbd_string是一个字符串,而不是字符串列表
    mypol = Polymer.from_pdb_string(pdb_str, chem_templates, mk_prep, allow_bad_res=allow_bad_res, default_altloc="A")
    rigid_pdbqt_string, flex_pdbqt_string = PDBQTWriterLegacy.write_string_from_polymer(mypol)
    

    with open(output_pdbqt, "w") as f:
        f.write(rigid_pdbqt_string) 



def run_Zn_protprepared(output_pdbqt, zn_pdbqt) -> None:
    args = [
        "-r", str(output_pdbqt),
        "-o", str(zn_pdbqt),
            ]
    run_mgltools_script(ZINC_PSEUDO, args, Path(zn_pdbqt).parent)
    # if not any(line.startswith(("ATOM", "HETATM")) and line.split()[-1] == "TZ"
    #            for line in Path(zn_pdbqt).read_text().splitlines()):
    #     raise ValueError(f"No TZ pseudo atom was added to {zn_pdbqt}; check the receptor zinc site")



def run_Zn_gpf(zn_pdbqt, gpf_zn_path, box_center, box_size, ligand_types, parameter_file):
    """Generate one GPF for a fixed receptor, box, and set of ligand atom types."""
    # if not ligand_types:
    #     raise ValueError("Ligand atom types are required to generate affinity maps")
    # parameter_types = {
    #     line.split()[1] for line in Path(parameter_file).read_text().splitlines()
    #     if line.startswith("atom_par ")
    # }
    # missing = set(ligand_types) - parameter_types
    # if missing:
    #     raise ValueError(f"Ligand types missing from {parameter_file}: {', '.join(sorted(missing))}")
    npts = box_size_to_npts(box_size)  # AutoGrid4 的 GPF 用 npts 和 spacing 表示网格尺寸。
    args = [
        "-r", str(zn_pdbqt),
        "-o", str(gpf_zn_path),
        # MGLTools pythonsh expands $@ unquoted; commas keep one -p value intact.
        "-p", "ligand_types=" + ",".join(ligand_types),
        "-p", f"npts={npts[0]},{npts[1]},{npts[2]}",
        "-p", f"gridcenter={box_center[0]:.3f},{box_center[1]:.3f},{box_center[2]:.3f}",
        "-p", f"parameter_file={Path(parameter_file).resolve()}",
    ]
    run_mgltools_script(PREPARE_GPF4ZN, args, Path(gpf_zn_path).parent)
    if not Path(gpf_zn_path).is_file() or Path(gpf_zn_path).stat().st_size == 0:
        raise RuntimeError(f"prepare_gpf4zn.py did not create {gpf_zn_path}")



def run_grid(receptor_pdb, output_dir, *,
             reference_lig_file=None, box_center=None, box_size=None,
             padding=5.0, charge_model=None, allow_bad_res=False,
             binary=None, gpf_only=False):
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

    output_zn_pdbqt = output_dir/ f'{base}_zn.pdbqt'
    print(f"Preparing receptor: {receptor_pdb.name}", file=sys.stderr)
    print(f"Box center: ({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f})", file=sys.stderr)
    print(f"Box size:   ({box_size[0]:.3f}, {box_size[1]:.3f}, {box_size[2]:.3f})", file=sys.stderr)
    

    prepare_receptor(receptor_pdb, output_pdbqt, charge_model, allow_bad_res)
    run_Zn_protprepared(output_pdbqt, output_zn_pdbqt)

    # Zn状态下的gpf
    print(f"Running ZN grid4: {output_zn_pdbqt}", file=sys.stderr)
    gpf_zn_path =  output_zn_pdbqt.with_suffix('.gpf')
    parameter_file = AD4ZN_PARAMETERS
    run_Zn_gpf(output_zn_pdbqt, gpf_zn_path, box_center, box_size, any_lig_base_types,
                 parameter_file=parameter_file)

    glg_path = gpf_zn_path.with_suffix(".glg")
    
    
    
    
    
    # get_grid
    if not gpf_only:
        grid_binary = binary or AUTOGRID4_BINARY
        version = subprocess.run([str(grid_binary), "--version"], capture_output=True, text=True, check=False)
        if "AutoGrid 4.2.6" in version.stdout + version.stderr:
            raise RuntimeError(
                "The GPF was generated, but AutoGrid 4.2.6 cannot correctly handle "
                "AutoDock4Zn nbp_r_eps. Use --gpf-only, or --binary with ADFR AutoGrid 4.2.7."
            )
        print(f"Running autogrid4: {gpf_zn_path.name}", file=sys.stderr)
        run_command([grid_binary, "-p", gpf_zn_path, "-l", glg_path],
                    gpf_zn_path.parent, "autogrid4")

    print(f"OK\t{base}\t{output_zn_pdbqt}")
    print(f"receptor_gpf\t{gpf_zn_path}", file=sys.stderr)
    if not gpf_only:
        print(f"grid_log\t{glg_path}", file=sys.stderr)
    return output_zn_pdbqt, gpf_zn_path, None if gpf_only else glg_path


def main(args):
    has_ref = args.ref_ligand_file is not None or args.ref_ligand_dir is not None
    has_box = args.box_center is not None or args.box_size is not None
    if has_ref == has_box:
        raise ValueError("Use either a reference ligand or --box-center with --box-size")
    if (args.box_center is None) != (args.box_size is None):
        raise ValueError("--box-center and --box-size must be used together")
    if (args.receptor_pdb is None) == (args.receptor_dir is None):
        raise ValueError("Provide exactly one of --receptor-pdb or --receptor-dir")

    if args.receptor_pdb:
        if args.ref_ligand_dir is not None:
            raise ValueError("Use --ref-ligand-file with --receptor-pdb")
        receptor_pdb = Path(args.receptor_pdb).resolve()
        output_dir = (args.output_dir or receptor_pdb.parent).resolve() / receptor_pdb.stem
        run_grid(
            receptor_pdb, output_dir,
            reference_lig_file=args.ref_ligand_file.resolve() if args.ref_ligand_file else None,
            box_center=parse_triplet(args.box_center),
            box_size=parse_triplet(args.box_size),
            padding=args.padding,
            charge_model=args.charge_model,
            allow_bad_res=args.allow_bad_res,
            binary=Path(args.binary).resolve() if args.binary else None,
            gpf_only=args.gpf_only,
        )

    else:
        if args.ref_ligand_file is not None:
            raise ValueError("Use --ref-ligand-dir with --receptor-dir")
        if args.data_list is None:
            raise ValueError("--data_list is required with --receptor-dir")
        for pdb_id in tqdm(load_idx_list(args.data_list)):
            ref_ligand_file = (args.ref_ligand_dir / pdb_id / f"{pdb_id}_ligand.sdf").resolve() if args.ref_ligand_dir else None
            receptor_pdb = (args.receptor_dir / f"{pdb_id}_protein_pocket10.pdb").resolve()
            output_dir = (args.output_dir or args.receptor_dir).resolve() / receptor_pdb.stem
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
                    gpf_only=args.gpf_only,
                )
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}", file=sys.stderr)


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
    parser.add_argument("--output-dir",       type=Path, default=None, help="Output directory. Default: receptor directory.")
    parser.add_argument("--allow-bad-res",    action="store_true")
    parser.add_argument("--charge-model",     choices=("gasteiger", "espaloma", "zero", "read"), default=None)
    parser.add_argument("--binary",           type=Path, default=None, help="autogrid4 binary path.")
    parser.add_argument("--gpf-only",         action="store_true", help="Generate the zinc receptor and GPF without running AutoGrid.")
    main(parser.parse_args())

    # receptor_pdb = /data3/lpw/dock_workdir/autodock_gpu/cov200_test/pocket/5REJ_protein_pocket10.pdb
    # ref-ligand_file = /data3/lpw/data/covdocker/data/processed/bonded/5REJ/5REJ_ligand.sdf
    # ref-ligand_dir = /data3/lpw/data/covdocker/data/processed/bonded
    # receptor_dir = /data3/lpw/dock_workdir/autodock_gpu/cov200_test/pocket


# 示例文件
# python -m autodockZn.get_grid \
#   --receptor-pdb /home/lpw/dock_scripts/autodock/autodockZn/test/proteinH.pdb \
#   --ref-ligand-file /home/lpw/dock_scripts/autodock/autodockZn/test/1s63_ligand.sdf \
#   --output-dir /home/lpw/dock_scripts/autodock/autodockZn/test/out
# Or replace --ref-ligand-file with --box-center 18 134 -1 --box-size 15 15 15.
