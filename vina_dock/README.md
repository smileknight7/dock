# AutoDock Vina Workflow

这个目录是一套面向 `AutoDock Vina` 的轻量工作流，当前脚本分成 4 步：

1. `get_pocket.py`
   从全蛋白按参考配体裁 pocket PDB
2. `prepare_receptor.py`
   把 receptor PDB 转成 `receptor.pdbqt`，并生成 Vina box 配置
3. `prepare_ligand.py`
   把单个 ligand、ligand 目录、或 `CSV + SMILES` 批量转成 `ligand.pdbqt`
4. `run_vina.py`
   对已经准备好的 `ligand.pdbqt` 运行 docking

当前实现基于：

- `RDKit`
- `Meeko`
- `Vina Python API`


## 文件说明

- `get_pocket.py`
  - 输入参考配体坐标和全蛋白 PDB
  - 输出 pocket PDB 和一个 summary 文本
  - 这一步是可选的，但通常建议先做

- `prepare_receptor.py`
  - 输入全蛋白 PDB 或 pocket PDB
  - 用 Meeko Python API 生成 `receptor.pdbqt`
  - 根据参考配体或手动参数写出 box
  - 输出：
    - `*.pdbqt`
    - `*.box.json`
    - `*.vina.txt`
    - 可选 `*.prepared.pdb`

- `prepare_ligand.py`
  - 支持：
    - `--ligand-file`
    - `--ligand-dir`
    - `--csv`
  - 单分子模式可直接指定 `--output-pdbqt`
  - 批量模式写到 `--output-dir`

- `run_vina.py`
  - 只接收已经准备好的 `.pdbqt`
  - 支持：
    - `--ligand-pdbqt`
    - `--ligand-dir`
  - 支持并行和简单进度条
  - 输出每个 ligand 的 pose 和总体 `results.csv`

- `vina_utils.py`
  - 共用工具函数
  - 包括：
    - RDKit 分子读取与 3D 规范化
    - Meeko ligand/receptor preparation
    - box 计算
    - CSV 读取
    - ligand spec 收集


## 环境要求

当前脚本依赖：

- `vina`
- `meeko`
- `rdkit`
- `numpy`
- `tqdm`


## 推荐流程

### 1. 可选：先裁 pocket

如果你不想直接拿全蛋白做 receptor preparation，推荐先裁 pocket：

```bash
python vina/get_pocket.py \
  --ligand-file /path/to/reference_ligand.sdf \
  --protein-file /path/to/protein.pdb \
  --output-dir /path/to/work/pocket \
  --output-basename protein_pocket10 \
  --pocket-cutoff 10.0
```

输出通常是：

- `/path/to/work/pocket/protein_pocket10.pdb`
- `/path/to/work/pocket/protein_pocket10.summary.txt`

说明：

- `get_pocket.py` 只做裁 pocket，不做 `pdbqt` 转换
- `Vina` 不要求必须裁 pocket
- 但如果全蛋白很大、杂质很多、链很多，先裁 pocket 往往更稳


### 2. 准备 receptor

可以直接吃全蛋白，也可以吃上一步裁好的 pocket PDB。

根据参考配体自动推 box：

```bash
python vina/prepare_receptor.py \
  --receptor-file /path/to/work/pocket/protein_pocket10.pdb \
  --reference-ligand /path/to/reference_ligand.sdf \
  --output-dir /path/to/work/receptor \
  --output-basename receptor
```

手动指定 box：

```bash
python vina/prepare_receptor.py \
  --receptor-file /path/to/protein_or_pocket.pdb \
  --box-center 10 11 12 \
  --box-size 20 20 20 \
  --output-dir /path/to/work/receptor \
  --output-basename receptor
```

主要输出：

- `receptor.pdbqt`
- `receptor.box.json`
- `receptor.vina.txt`

可选输出：

- `receptor.prepared.pdb`

注意：

- `prepare_receptor.py` 不负责裁 pocket
- 它只负责 receptor preparation 和 box 配置
- 水、金属、辅因子、共晶配体要不要保留，仍然需要你在输入 PDB 阶段决定


### 3. 准备 ligand

#### 单个 ligand 文件

```bash
python vina/prepare_ligand.py \
  --ligand-file /path/to/ligand.sdf \
  --output-pdbqt /path/to/work/ligand.pdbqt
```

#### 目录批量

```bash
python vina/prepare_ligand.py \
  --ligand-dir /path/to/ligand_inputs \
  --output-dir /path/to/work/ligands \
  --jobs 4
```

#### CSV + SMILES 批量

```bash
python vina/prepare_ligand.py \
  --csv /path/to/ligands.csv \
  --smiles-column smiles \
  --id-column ligand_id \
  --output-dir /path/to/work/ligands \
  --jobs 4
```

CSV 最少需要：

- `smiles`

可选：

- `ligand_id`

示例：

```csv
ligand_id,smiles
ethanol,CCO
aspirin,CC(=O)Oc1ccccc1C(=O)O
```

支持的 ligand 输入格式：

- `.sdf`
- `.mol2`
- `.mol`
- `.pdb`
- `.pdbqt`

如果输入不是 `.pdbqt`：

- 先用 RDKit 补氢
- 如果没有 3D 构象则自动生成
- 再用 Meeko 转成 `ligand.pdbqt`


### 4. 运行 docking

#### 单个 prepared ligand

```bash
python vina/run_vina.py \
  --receptor-pdbqt /path/to/work/receptor/receptor.pdbqt \
  --box-config /path/to/work/receptor/receptor.box.json \
  --ligand-pdbqt /path/to/work/ligand.pdbqt \
  --output-dir /path/to/work/docking_single
```

#### 目录批量 docking

```bash
python vina/run_vina.py \
  --receptor-pdbqt /path/to/work/receptor/receptor.pdbqt \
  --box-config /path/to/work/receptor/receptor.box.json \
  --ligand-dir /path/to/work/ligands \
  --output-dir /path/to/work/docking \
  --jobs 4
```

如果不想用 `box.json`，也可以直接手动给：

```bash
python vina/run_vina.py \
  --receptor-pdbqt /path/to/work/receptor/receptor.pdbqt \
  --box-center 10 11 12 \
  --box-size 20 20 20 \
  --ligand-dir /path/to/work/ligands \
  --output-dir /path/to/work/docking \
  --jobs 4
```


## 输出结构

### get_pocket.py

```text
pocket/
  protein_pocket10.pdb
  protein_pocket10.summary.txt
```

### prepare_receptor.py

```text
receptor/
  receptor.pdbqt
  receptor.box.json
  receptor.vina.txt
  receptor.prepared.pdb   # 只有 --write-pdb 时才有
```

### prepare_ligand.py 批量模式

```text
ligands/
  ligand_a/
    ligand_a.pdbqt
  ligand_b/
    ligand_b.pdbqt
```

### run_vina.py

```text
docking/
  results.csv
  ligand_a/
    ligand_a_out.pdbqt
    summary.json
  ligand_b/
    ligand_b_out.pdbqt
    summary.json
```

`run_vina.py` 默认不再重复做 ligand preparation，所以它只写 docking 结果，不再复制或重写输入 `ligand.pdbqt`。


## results.csv 字段

- `ligand_id`
- `status`
- `best_score`
- `pose_count`
- `poses_pdbqt`
- `ligand_pdbqt`
- `source`
- `error`

状态含义：

- `ok`
  - docking 成功
- `skipped`
  - 开启了 `--skip-existing`，复用了已有结果
- `failed`
  - 当前 ligand 失败


## 常用参数

### get_pocket.py

- `--pocket-cutoff`
  - 以参考配体为中心，选择多少 A 内的残基进入 pocket
- `--box-padding`
  - pocket summary 里建议 box 的额外 padding
- `--keep-hetero`
  - 是否保留非标准杂原子残基

### prepare_receptor.py

- `--reference-ligand`
  - 自动从参考配体推 box
- `--box-center / --box-size`
  - 手动指定 box
- `--padding`
  - 自动 box 时的外扩距离
- `--allow-bad-res`
  - 允许删除缺失原子的残基
- `--write-pdb`
  - 额外写 prepared receptor PDB

### prepare_ligand.py

- `--jobs`
  - batch preparation 并发数
- `--skip-existing`
  - 跳过已存在的 ligand 结果
- `--charge-model`
  - 传给 Meeko
- `--rigid-macrocycles`
  - 保持大环刚性

### run_vina.py

- `--jobs`
  - ligand 级并发数
- `--cpu`
  - 每个 docking 任务内部使用的 CPU 线程数
- `--exhaustiveness`
  - 搜索强度
- `--n-poses`
  - 输出 pose 数量
- `--energy-range`
  - 保留 pose 的能量窗口
- `--skip-existing`
  - 跳过已完成的 ligand
- `--no-recursive`
  - `--ligand-dir` 下只扫描当前层

`run_vina.py` 里已经带了简单进度条。  
当 ligand 数量大于 1 且输出连接到交互终端时，会自动显示 `tqdm` 进度条。


## 一个完整示例

先裁 pocket：

```bash
python vina/get_pocket.py \
  --ligand-file /data3/lpw/dock_workdir/autodock/test/3I03_ligand.sdf \
  --protein-file /data3/lpw/dock_workdir/autodock/test/3I03_protein.pdb \
  --output-dir /tmp/vina_example/pocket \
  --output-basename 3I03_pocket10
```

再准备 receptor：

```bash
python vina/prepare_receptor.py \
  --receptor-file /tmp/vina_example/pocket/3I03_pocket10.pdb \
  --reference-ligand /data3/lpw/dock_workdir/autodock/test/3I03_ligand.sdf \
  --output-dir /tmp/vina_example/receptor \
  --output-basename receptor
```

再准备 ligand：

```bash
python vina/prepare_ligand.py \
  --csv /path/to/ligands.csv \
  --smiles-column smiles \
  --id-column ligand_id \
  --output-dir /tmp/vina_example/ligands \
  --jobs 4
```

最后跑 docking：

```bash
python vina/run_vina.py \
  --receptor-pdbqt /tmp/vina_example/receptor/receptor.pdbqt \
  --box-config /tmp/vina_example/receptor/receptor.box.json \
  --ligand-dir /tmp/vina_example/ligands \
  --output-dir /tmp/vina_example/docking \
  --jobs 4
```


## 注意事项

- 这是 Vina 流程，不需要 `gpf`、`autogrid4`、`dpf`
- `prepare_receptor.py` 不做口袋裁剪，口袋裁剪由 `get_pocket.py` 负责
- 是否必须裁 pocket 取决于你的体系；Vina 关键是 box 合理
- 如果全蛋白很大、杂质很多、链很多，建议先裁 pocket 再准备 receptor
