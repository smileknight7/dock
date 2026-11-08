# 出现的问题



## FAIL: for residue_key='A:462', 3 have passed: ['HIE', 'HID', 'HIP'] and tied for fewest missing and excess H: HIE HID 

```

这个有两种解决方式， 一个是使用openmm 按照PH去补充H，一种是，直接使用meeko按默认模板重新补氢，

# 方法1
from openmm.app import PDBFile, Modeller, element

pdb = PDBFile("data/3A1D/3A1D_protein.pdb")
modeller = Modeller(pdb.topology, pdb.positions)

# 重新选择质子化状态，避免原有氢限制自动判断。
modeller.delete([
    atom for atom in modeller.topology.atoms()
    if atom.element == element.hydrogen
])
modeller.addHydrogens(pH=7.4)  # 按实际实验条件设置

with open("data/3A1D/3A1D_protein_H.pdb", "w") as f:
    PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)


# 方法2 直接对原本脚本进行去H处理


python - <<'PY'
from pathlib import Path

src = Path("data/3A1D/3A1D_protein.pdb")
dst = src.with_name("3A1D_protein_noH.pdb")

lines = src.read_text().splitlines(keepends=True)
dst.write_text("".join(
    line for line in lines
    if not (
        line.startswith(("ATOM  ", "HETATM"))
        and line[76:78].strip() in {"H", "D"}
    )
))
print(dst)
PY



```