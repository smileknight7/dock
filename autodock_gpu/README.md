# AutoDock-GPU Skeleton

这个目录现在承载 GPU 专用的前处理和运行脚本，默认工作流对齐到：

- 可选：从全蛋白裁 pocket PDB
- Meeko 准备 ligand
- Meeko 准备 receptor 并生成 `gpf`
- `autogrid4` 生成 `.maps.fld` 和各类 `.map`
- AutoDock-GPU direct mode 读取 `ligand.pdbqt + receptor.maps.fld`

## 目录约定

- `config.py`
  - 固定本机 `AutoDock-GPU` 可执行文件默认路径
  - 提供默认 `device`、`nrun`、`nev`
- `prepare_ligand.py`
  - 用 Meeko 生成 ligand `.pdbqt`
  - 支持单文件和目录批处理
  - 在调用 Meeko 前先用 RDKit 显式补氢，规避常见输入兼容问题
- `get_pocket.py`
  - 从全蛋白 PDB 按参考配体坐标裁出 pocket PDB
  - 输出 pocket summary，记录建议的 `grid_center` 和 `box_size`
  - 不负责 PDBQT 转换，下一步交给 `prepare_receptor.py`
- `prepare_receptor.py`
  - 用 Meeko 生成 receptor `.pdbqt` 和 `.gpf`
  - 支持 `--box_enveloping` 或手动 `--box_size/--box_center`
  - 支持按实际 ligand `PDBQT` 裁剪 `ligand_types/map`，避免 AutoGrid4 超过 14 个 ligand map 的上限
- `build_grid.py`
  - 调用 `autogrid4` 生成 `.glg/.maps.fld/*.map`
  - 支持单个 `gpf` 和批量 `gpf-dir`
- `run_gpu.py`
  - AutoDock-GPU 运行入口
  - 推荐使用 shared-grid direct mode：`ligand.pdbqt + receptor.maps.fld`
  - `--ligand-dir` 会递归扫描配体目录，并在输出目录中镜像子目录结构
  - `--filelist` 接受一个 ligand 列表文件，每行一个 `pdbqt` 路径；脚本会按这份列表调度 docking，并额外写出一份 AD-GPU 风格的 batch filelist 供检查
  - `import_dpf` 保留为兼容入口，不是推荐主流程

## 你现在已有的输入

推荐工作流：

```bash
/data3/lpw/conda/envs/autodock-gpu/bin/python /home/lpw/dock_scripts/autodock/autodock-GPU/get_pocket.py \
  --ligand-file /path/to/ligand.sdf \
  --protein-file /path/to/protein.pdb \
  --output-dir /path/to/work/pocket \
  --output-basename protein_pocket10
```

```bash
/data3/lpw/conda/envs/autodock-gpu/bin/python /home/lpw/dock_scripts/autodock/autodock-GPU/prepare_ligand.py \
  --ligand-file /path/to/ligand.sdf \
  --output-pdbqt /path/to/work/ligand.pdbqt
```

```bash
/data3/lpw/conda/envs/autodock-gpu/bin/python /home/lpw/dock_scripts/autodock/autodock-GPU/prepare_receptor.py \
  --receptor-file /path/to/work/pocket/protein_pocket10.pdb \
  --reference-ligand /path/to/ligand.sdf \
  --ligand-pdbqt /path/to/work/ligand.pdbqt \
  --output-dir /path/to/work/receptor \
  --output-basename receptor
```

```bash
python /home/lpw/dock_scripts/autodock/autodock-GPU/build_grid.py \
  --gpf /path/to/work/receptor/receptor.gpf
```

```bash
python /home/lpw/dock_scripts/autodock/autodock-GPU/run_gpu.py \
  --ligand /path/to/work/ligand.pdbqt \
  --fld /path/to/work/receptor/receptor.maps.fld \
  --output-dir /path/to/work/docking
```

批量 shared-grid 运行：

```bash
python /home/lpw/dock_scripts/autodock/autodock-GPU/run_gpu.py \
  --ligand-dir /path/to/work/ligands \
  --fld /path/to/work/receptor/receptor.maps.fld \
  --output-dir /path/to/work/docking \
  --jobs 4
```

如果 `prepare_ligand.py` 产物放在多级子目录下，`run_gpu.py` 会递归扫描 `*.pdbqt`，并把 `dlg/xml` 写回到 `output-dir` 下对应的相对目录。

如果你已经有一个 ligand 路径列表文件，也可以直接用 `--filelist`：

```bash
python /home/lpw/dock_scripts/autodock/autodock-GPU/run_gpu.py \
  --filelist /path/to/ligands.txt \
  --fld /path/to/work/receptor/receptor.maps.fld \
  --output-dir /path/to/work/docking
```

这里的 `ligands.txt` 是 wrapper 级别的简单列表文件，每行一个 ligand `pdbqt` 路径，允许相对路径和绝对路径。脚本会按这份列表运行 docking，并在 `output-dir` 下额外生成一份 AD-GPU 风格的 batch filelist。

## 验证产物

当前仓库已经在下面路径完成了一次实跑验证：

- `/data3/lpw/dock_workdir/autodock/dock_gpu/meeko_validate/ligand`
- `/data3/lpw/dock_workdir/autodock/dock_gpu/meeko_validate/receptor`
- `/data3/lpw/dock_workdir/autodock/dock_gpu/meeko_validate/docking`

验证顺序是：

- `prepare_ligand.py`
- `prepare_receptor.py --ligand-pdbqt ...`
- `build_grid.py`
- `run_gpu.py --ligand ... --fld ...`
