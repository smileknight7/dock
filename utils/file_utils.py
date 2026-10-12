import rdkit 
from rdkit import Chem
from pathlib import Path
from meeko import MoleculePreparation, PDBQTWriterLegacy
import pandas as pd
from biopandas.pdb import PandasPdb


# 按照id从dir中读
def load_idx_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line[:-1] if line[-1] == '\n' else line for line in lines]
    return lines


def read_string_fromtxt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line[:-1] if line[-1] == '\n' else line for line in lines]
    return lines

# 直接从dir中读
def collect_ligand_files(ligand_dir):
    return [(f.stem, str(f)) for f in Path(ligand_dir).iterdir()
            if f.suffix.lower() in {".sdf", ".mol2", ".mol", ".pdb"}]

# 直接从smiles中读
def read_smiles_records(csv_path, id_column=None, smiles_column="SMILES"):
    df = pd.read_csv(csv_path)
    items = [(str(row[id_column]) if id_column else f"ligand_{i:04d}",
              str(row[smiles_column])) for i, row in df.iterrows()]
    return [(ligand_id, smi) for ligand_id, smi in items if smi.strip()]


# 写pdbqt
def write_pdbqt(mol, output_file):
    mk_prep = MoleculePreparation()
    mol_status = mk_prep.prepare(mol)[0]
    text, ok, err = PDBQTWriterLegacy.write_string(mol_status)
    if not ok:
        raise RuntimeError(f"Meeko PDBQT write failed: {err}")
    Path(output_file).write_text(text)

# pandasdb -->pdb
def write_pocket_pdb(df_pocket, output_path):
    out = PandasPdb()
    out.df['ATOM'] = df_pocket[df_pocket['record_name'] == 'ATOM'].reset_index(drop=True)
    out.df['HETATM'] = df_pocket[df_pocket['record_name'] == 'HETATM'].reset_index(drop=True)
    out.to_pdb(str(output_path), records=['ATOM', 'HETATM'])


def load_mol(source, is_smiles=False):                      # 这里可以加上一个兼容性检查？没有sdf就使用别的格式？
    if is_smiles:
        mol = Chem.MolFromSmiles(source)
    else:
        suffix = Path(source).suffix.lower()
        if suffix == ".sdf":
            mol = Chem.SDMolSupplier(source, removeHs=False)[0]
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(source, removeHs=False)
        elif suffix == ".mol":
            mol = Chem.MolFromMolFile(source, removeHs=False)
        elif suffix == ".pdb":
            mol = Chem.MolFromPDBFile(source, removeHs=False)
        else:
            raise ValueError(f"Unsupported format: {source}")
    if mol is None:
        raise ValueError(f"RDKit failed to load: {source}")
    return mol


