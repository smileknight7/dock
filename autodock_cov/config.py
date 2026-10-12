
from pathlib import Path
import subprocess

'''
共价文件部分

ligcovalent.pdb                 原始ligand文件
3upo_protein.pdbqt              原始pocket文件
3upo_protein_rigid.pdbqt        pocket刚性部分
3upo_protein_flex.pdbqt         pocket柔性部分
ligcovalent_flex.pdbqt          ligand + pocket柔性部分

ligcovalent_3upo_protein.dlg    对接结果文件               



'''


AA_ATOMS = {'N','CA','C','O','OXT','CB','SG','OG','OG1','OH',
             'ND1','NE2','OD1','OD2','OE1','OE2','NZ','SD'}




# mgltools路径配置
MGLTOOLS_ROOT = Path("/data3/lpw/software/autodock4/mgltools_x86_64Linux2_1.5.7")
MGLTOOLS_UTILS = MGLTOOLS_ROOT / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24"
PYTHONSH = MGLTOOLS_ROOT / "bin" / "pythonsh"

# 工具脚本路径配置
PREPARE_LIGAND4 = MGLTOOLS_UTILS / "prepare_ligand4.py"
PREPARE_ALTCONFS = MGLTOOLS_UTILS / "prepare_pdb_split_alt_confs.py"
PREPARE_RECEPTOR4 = MGLTOOLS_UTILS / "prepare_receptor4.py"
PREPARE_FLEX_RECEPTOR4 = MGLTOOLS_UTILS / "prepare_flexreceptor4.py"
PREPARE_GPF4 = MGLTOOLS_UTILS / "prepare_gpf4.py"
PREPARE_DPF4 = MGLTOOLS_UTILS / "prepare_dpf4.py"
PDBQT_TO_PDB = MGLTOOLS_UTILS / "pdbqt_to_pdb.py"
GET_COMPLEX = MGLTOOLS_UTILS / "write_all_complexes.py"
GET_CONFORMATIONS = MGLTOOLS_UTILS / "write_conformations_from_dlg.py"
GET_LOWEST_CONFORMATION = MGLTOOLS_UTILS / "write_lowest_energy_ligand.py"     # -f result.dlg -o result.pdbqt

# 共价对接工具
AdCovScriptRoot = Path('/data3/lpw/software/autodock4/adCovalentDockResidue/adcovalent')
PREPARECOVALENT = AdCovScriptRoot / "prepareCovalent.py"


# 功能脚本
AUTOGRID4 = Path("/data3/lpw/software/autodock4/x86_64Linux2/autogrid4")
AUTODOCK4 = Path("/data3/lpw/software/autodock4/x86_64Linux2/autodock4")

GRID_SPACING = 0.375

def get_script_help(script: Path) -> str:
    if not PYTHONSH.is_file():
        raise FileNotFoundError(f"pythonsh不存在: {PYTHONSH}")

    if not script.is_file():
        raise FileNotFoundError(f"工具脚本不存在: {script}")

    result = subprocess.run(
        [str(PYTHONSH), str(script), "-h"],
        cwd=script.parent,
        text=True,
        capture_output=True,
    )
    return result



def run_command(cmd: list[str | Path], cwd: Path, label: str = "command") -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )

    if result.returncode:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"{label} failed:\n{msg}")




def run_prepare_ligand4(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, PREPARE_LIGAND4, *args],
        cwd,
        label=PREPARE_LIGAND4.name,
    )

def run_prepare_altconfs(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, PREPARE_ALTCONFS, *args],
        cwd,
        label=PDBQT_TO_PDB.name,
    )


def run_prepare_receptor4(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, PREPARE_RECEPTOR4, *args],
        cwd,
        label=PREPARE_RECEPTOR4.name,
    )


def run_prepare_gpf4(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, PREPARE_GPF4, *args],
        cwd,
        label=PREPARE_GPF4.name,
    )


def run_prepare_dpf4(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, PREPARE_DPF4, *args],
        cwd,
        label=PREPARE_DPF4.name,
    )


def run_pdbq2pdb(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, PDBQT_TO_PDB, *args],
        cwd,
        label=PDBQT_TO_PDB.name,
    )


def run_getcomplex(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, GET_COMPLEX, *args],
        cwd,
        label=GET_COMPLEX.name,
    )


def run_get_conformations(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, GET_CONFORMATIONS, *args],
        cwd,
        label=GET_CONFORMATIONS.name,
    )


def run_mgltools_script(
    script: Path,
    args: list[str | Path],
    cwd: Path,
) -> None:
    run_command([PYTHONSH, script, *args], cwd, label=script.name)


def run_autogrid4(gpf: Path, glg: Path) -> None:
    run_command([AUTOGRID4, "-p", gpf, "-l", glg], gpf.parent, "autogrid4")


def run_autodock4(dpf: Path, dlg: Path) -> None:
    run_command([AUTODOCK4, "-p", dpf, "-l", dlg], dpf.parent, "autodock4")

# 共价部分docking

def run_prepare_covalent(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, PREPARECOVALENT, *args],
        cwd,
        label=PREPARECOVALENT.name,
    )


def run_prepare_flex_receptor4(args: list[str | Path], cwd: Path) -> None:
    run_command(
        [PYTHONSH, PREPARE_FLEX_RECEPTOR4, *args],
        cwd,
        label=PREPARE_FLEX_RECEPTOR4.name,
    )


if __name__ == "__main__":
    help = get_script_help(PREPARE_DPF4)
    print(help.stdout)




    '''
    检查结果如下
    Usage: write_all_complexes.py -d directory

        Description of command...
            -d     docking filename
            -r     receptor filename
        Optional parameters:
            [-b]    write complex of best result and receptor only (default is write all)
            [-o]    output file stem
            (default is receptorname_ligandname plus '_confnum.pdbqt')
            [-v]    verbose output

        
    '''
