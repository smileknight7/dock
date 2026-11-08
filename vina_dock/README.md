# AutoDock Vina

在仓库根目录运行，使用 `/data3/lpw/conda/envs/autodock-gpu/bin/python`。入口依次是：

1. `python -m vina_dock.prepare_ligand --ligand-file <ligand.sdf> --output-dir <ligands>`
2. `python -m vina_dock.prepare_receptor --receptor-file <protein.pdb> --reference-ligand <reference.sdf> --output-dir <receptor>`
3. `python -m vina_dock.run_vina --receptor-pdbqt <receptor.pdbqt> --box-config <receptor.box.json> --ligand-pdbqt <ligand.pdbqt> --output-dir <docking>`
4. `python -m vina_dock.analysis_result --input-dir <docking> --output-dir <analysis>`

可选用 `python -m utils.get_pocket` 先裁口袋。受体也支持手动 `--box-center X Y Z --box-size X Y Z`。配体准备支持 `--ligand-dir` 或 `--ligand-csv` 批量输入；对接支持 `--ligand-dir` 批量输入。当前 Vina 入口提供 `vina` 与 `vinardo` 评分；Zn 的 AD4 map 评分在 `autodockZn.docking` 中。完整参数见各脚本 `--help`。
