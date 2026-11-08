# 结果分析入口

从仓库根目录以 `python -m` 运行，指定 `--input-dir` 和 `--output-dir`。

| 结果格式 | 推荐入口 | 主要输出 |
|---|---|---|
| AutoDock4 `.dlg` | `analysis.analysis_result` | 全部配体打分 CSV、top-N 文本，可导出最优 PDB pose |
| AutoDock-GPU `.dlg`（Meeko 配体） | `analysis.analysis_result2` | 每次 run 的 CSV、配体汇总 CSV；有参考 SDF 时计算对称修正的重原子 RMSD，可导出 SDF poses |
| Vina / AutoDock4Zn `*_out.pdbqt` | `vina_dock.analysis_result` | 全部配体打分 CSV、top-N 文本，可导出最优 PDB pose |

例如 GPU 的 RMSD 分析：

```bash
python -m analysis.analysis_result2 \
  --input-dir /path/to/docking \
  --output-dir /path/to/analysis \
  --gt-ligand-dir /path/to/reference_ligands \
  --export-poses
```

参考配体目录格式为 `<id>/<id>_ligand.sdf`，其中 `<id>` 与 DLG 文件名一致。`rmsd_analysis.py` 是旧的 GPU 专用单指标脚本，只适用于该目录格式；它不能替代三个推荐入口。现在它只在显式执行并提供 `--data-dir`、`--result-dir` 时运行。
