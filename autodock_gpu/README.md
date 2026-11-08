# AutoDock-GPU

在仓库根目录运行，使用 `/data3/lpw/conda/envs/autodock-gpu/bin/python`。入口依次是：

1. `python -m autodock_gpu.ligand_prepare --ligand-file <ligand.sdf> --output-dir <ligands>`
2. `python -m autodock_gpu.get_grid --receptor-pdb <protein.pdb> --ref-ligand-file <reference.sdf> --output-dir <grid>`
3. `python -m autodock_gpu.docking --ligand <ligand.pdbqt> --fld <grid>/<receptor>/<receptor>.maps.fld --output-dir <docking> --device 1`
4. `python -m analysis.analysis_result2 --input-dir <docking> --output-dir <analysis>`

受体有不完整残基时，可给网格步骤加 `--allow-bad-res`。批量共享网格用 `docking --ligand-dir <dir> --fld <file>`；按配体/受体 ID 配对用 `--ligand-dir <dir> --grid-dir <dir> --data-list <file>`。`--num_workers` 默认 1，单张 GPU 上建议保持 1。二进制依赖的 CUDA 11 动态库从 `--conda-prefix` 指定的环境加载。完整参数见各脚本 `--help`。

`--dpf` / `--dpf-dir` 只透传给 AutoDock-GPU 的 `--import_dpf`。当前 MGLTools 生成的标准 DPF 包含二进制不支持的 `autodock_parameter_version`，本机实测无法直接导入；请使用上述 `.maps.fld` 模式。
