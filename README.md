# AutoDock4 Workflow

这个目录现在包含两套流程：

- `autodock/`：AutoDock4 工作流，保留 `MGLTools` 做 ligand/receptor preparation
- `vina/`：AutoDock Vina 工作流
- `autodock-GPU/`：AutoDock-GPU 相关脚本

本文档只说明 `autodock/` 这套 AutoDock4 脚本。

## 脚本结构

`autodock/` 目录下当前的入口脚本如下：

- `config.py`
  - 集中定义 `MGLTOOLS_ROOT`、`pythonsh`、`prepare_ligand4.py`、`prepare_receptor4.py`、`prepare_gpf4.py`、`prepare_dpf4.py`、`autogrid4`、`autodock4`
- `ligand_prepare.py`
  - ligand 预处理，输入原始 ligand 文件或 `CSV + SMILES`，输出 `ligand.pdbqt`
- `get_pocket.py`
  - 参考配体裁 pocket，并把 pocket 转成 `pocket.pdbqt`
- `get_grid.py`
  - 生成 `.gpf`、`.dpf`，并运行 `autogrid4`
- `docking.py`
  - 运行 `autodock4`
- `analysis_result.py`
  - 解析 `.dlg`，汇总打分结果

## 依赖说明

这套流程保留 `MGLTools` 作为 AutoDock4 标准前处理工具：

- ligand PDB -> PDBQT：`prepare_ligand4.py`
- receptor PDB -> PDBQT：`prepare_receptor4.py`
- grid 参数：`prepare_gpf4.py`
- docking 参数：`prepare_dpf4.py`

同时使用 `RDKit` 做以下工作：

- 从 `sdf/mol2/mol/pdb` 读取 ligand
- 从 `SMILES` 生成 3D 构象
- 读取参考配体坐标

路径集中写在 [autodock/config.py](/home/lpw/dock_scripts/autodock/autodock/config.py:1)。如果你的安装位置不同，只需要改这一个文件。

## 推荐流程

标准顺序：

1. `ligand_prepare.py`
2. `get_pocket.py`
3. `get_grid.py`
4. `docking.py`
5. `analysis_result.py`

对应 AutoDock4 原生链路：

`prepare_ligand4.py -> prepare_receptor4.py -> prepare_gpf4.py -> autogrid4 -> prepare_dpf4.py -> autodock4`

## 1. Ligand Preparation

### 单个 ligand

```bash
python autodock/ligand_prepare.py \
  --ligand-file /path/to/ligand.sdf \
  --output-pdbqt /path/to/work/ligand.pdbqt
```

也可以把结果写到目录里：

```bash
python autodock/ligand_prepare.py \
  --ligand-file /path/to/ligand.sdf \
  --output-dir /path/to/work/ligands
```

### 目录批量模式

```bash
python autodock/ligand_prepare.py \
  --ligand-dir /path/to/ligands \
  --output-dir /path/to/work/ligands_pdbqt \
  --jobs 4
```

支持的输入格式：

- `.sdf`
- `.mol2`
- `.mol`
- `.pdb`

### CSV + SMILES 批量模式

```bash
python autodock/ligand_prepare.py \
  --csv /path/to/ligands.csv \
  --smiles-column smiles \
  --id-column ligand_id \
  --output-dir /path/to/work/ligands_pdbqt \
  --jobs 4
```

常用参数：

- `--smiles-column`
- `--id-column`
- `--jobs`
- `--skip-existing`
- `--no-recursive`

## 2. Pocket Preparation

`get_pocket.py` 的职责是：

- 从全蛋白 `protein.pdb` 中截取结合口袋
- 输出 `pocket.pdb`
- 用 `prepare_receptor4.py` 生成 `pocket.pdbqt`
- 根据参考配体写出 `preprocess_summary.txt`

```bash
python autodock/get_pocket.py \
  --ligand-file /path/to/reference_ligand.sdf \
  --protein-file /path/to/protein.pdb \
  --output-dir /path/to/work/pocket \
  --output-basename receptor_pocket10 \
  --pocket-cutoff 10.0
```

输出通常包括：

- `receptor_pocket10.pdb`
- `receptor_pocket10.pdbqt`
- `preprocess_summary.txt`

`preprocess_summary.txt` 里会记录：

- `grid_center_xyz`
- `ligand_box_size_xyz`
- pocket 原子数和残基数

这些信息会被 `get_grid.py` 继续使用。

## 3. Grid Preparation

`get_grid.py` 做三件事：

- 调用 `prepare_gpf4.py` 生成 `.gpf`
- 调用 `prepare_dpf4.py` 生成 `.dpf`
- 调用 `autogrid4` 生成 maps 和 `.glg`

### 单任务模式

```bash
python autodock/get_grid.py \
  --ligand-pdbqt /path/to/ligand.pdbqt \
  --receptor-pdbqt /path/to/receptor_pocket10.pdbqt \
  --summary /path/to/preprocess_summary.txt \
  --output-dir /path/to/work/grid \
  --use-custom-grid-box
```

### 批量模式

```bash
python autodock/get_grid.py \
  --ligand-dir /path/to/ligands_pdbqt \
  --receptor-pdbqt /path/to/receptor_pocket10.pdbqt \
  --summary /path/to/preprocess_summary.txt \
  --batch-output-dir /path/to/work/grid_batch \
  --use-custom-grid-box \
  --skip-existing
```

说明：

- `--use-custom-grid-box` 会读取 `preprocess_summary.txt` 中的 `ligand_box_size_xyz`，换算为 `npts`
- 不加这个参数时，沿用 `prepare_gpf4.py` 的默认 box 设置
- 批量模式会为每个 ligand 建一个单独子目录

## 4. Docking

`docking.py` 支持单任务和批量模式。

### 单任务模式

```bash
python autodock/docking.py \
  --gpf /path/to/work/grid/receptor_pocket10.gpf \
  --dpf /path/to/work/grid/ligand_receptor_pocket10.dpf \
  --output-dir /path/to/work/docking \
  --skip-autogrid-if-glg-exists
```

### 批量模式

```bash
python autodock/docking.py \
  --dpf-dir /path/to/work/grid_batch \
  --batch-output-dir /path/to/work/docking_batch \
  --jobs 4 \
  --skip-autogrid-if-glg-exists \
  --skip-existing
```

说明：

- `--dpf-dir` 会递归扫描 `.dpf`
- 批量模式会先按 `.gpf` 去重，再并行运行 `autogrid4`
- 然后并行运行 `autodock4`
- `--skip-autogrid-if-glg-exists` 适合 maps 已经准备好的场景

输出主要是：

- `.dlg`
- 对应的 `.glg`

## 5. Result Analysis

`analysis_result.py` 用来解析 `.dlg`，统计最佳分数和聚类信息。

```bash
python autodock/analysis_result.py \
  --input-dir /path/to/work/docking_batch \
  --output-dir /path/to/work/analysis \
  --expected-runs 10
```

说明：

- `--input-dir` 和 `--output-dir` 现在都是必填
- 默认按 `AutoDock4` 常见的 `10` 次 run 校验
- 如果实际 run 数和预期不一致，会标成 `partial`

## 目录组织建议

一个清晰的工作目录可以类似这样：

```text
work/
  pocket/
    receptor_pocket10.pdb
    receptor_pocket10.pdbqt
    preprocess_summary.txt
  ligands_pdbqt/
    ligand_a.pdbqt
    ligand_b.pdbqt
  grid_batch/
    ligand_a/
    ligand_b/
  docking_batch/
    ligand_a/
    ligand_b/
  analysis/
```

## 已验证命令

下面这些命令已经在本地样本上跑通过：

```bash
python autodock/ligand_prepare.py \
  --ligand-file /data3/lpw/dock_workdir/autodock/test/3I03_ligand.sdf \
  --output-pdbqt /tmp/ad4_smoke/3I03_ligand.pdbqt
```

```bash
python autodock/get_pocket.py \
  --ligand-file /data3/lpw/dock_workdir/autodock/test/3I03_ligand.sdf \
  --protein-file /data3/lpw/dock_workdir/autodock/test/3I03_protein.pdb \
  --output-dir /tmp/ad4_smoke/pocket \
  --output-basename 3I03_pocket10 \
  --pocket-cutoff 10.0
```

```bash
python autodock/get_grid.py \
  --ligand-pdbqt /tmp/ad4_smoke/3I03_ligand.pdbqt \
  --receptor-pdbqt /tmp/ad4_smoke/pocket/3I03_pocket10.pdbqt \
  --summary /tmp/ad4_smoke/pocket/preprocess_summary.txt \
  --output-dir /tmp/ad4_smoke/grid_single \
  --use-custom-grid-box
```

```bash
python autodock/docking.py \
  --gpf /tmp/ad4_smoke/grid_single/3I03_pocket10.gpf \
  --dpf /tmp/ad4_smoke/grid_single/3I03_ligand_3I03_pocket10.dpf \
  --output-dir /tmp/ad4_smoke/docking_single \
  --skip-autogrid-if-glg-exists
```

```bash
python autodock/analysis_result.py \
  --input-dir /tmp/ad4_smoke/docking_single \
  --output-dir /tmp/ad4_smoke/analysis_single \
  --expected-runs 10
```

批量测试也已经验证过：

- `ligand_prepare.py --csv ... --jobs 2`
- `get_grid.py --ligand-dir ...`
- `docking.py --dpf-dir ... --jobs 2`
- `analysis_result.py` 对 batch 输出的解析

## 注意事项

- 这套流程依赖 `autodock/config.py` 里配置的二进制路径
- `get_pocket.py` 输出的是 pocket receptor，不是全蛋白 receptor
- `get_grid.py` 和 `docking.py` 的 batch 模式都会为每个 ligand 单独建目录
- 如果你已经手动准备好了 `.pdbqt`、`.gpf`、`.dpf`，也可以只跑后半段脚本
