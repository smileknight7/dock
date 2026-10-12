#!/usr/bin/env python3
"""Shared configuration for AutoDock-GPU helper scripts."""


from pathlib import Path


ADGPU_BINARY = Path("/data3/lpw/software/autodockgpu/adgpu-v1.6_linux_x64_cuda11_128wi")
AUTOGRID4_BINARY = Path("/data3/lpw/software/autodock4/x86_64Linux2/autogrid4")
CONDA_PREFIX = Path("/data3/lpw/conda/envs/autodock-gpu")

DEFAULT_DEVICE = 1
DEFAULT_NRUN = 100              # 独立 LGA run 数；heuristics 开启时每个 run 的评估预算会随它增大而缩小
DEFAULT_NEV = 12_000_000        # 每个 run 评估次数的上限；设得比 heuristics 的预算高，避免把它压低（原来的 250 万会压低 >=10 个可旋转键的配体）
DEFAULT_HEURISTICS = 1


any_lig_base_types = [
   "HD",
   "C",
   "A",
   "N",
   "NA",
   "OA",
   "F",
   "P",
   "SA",
   "S",
   "Cl",
   "Br",
   "I",
   # "Si",          当前这个版本的autograd好像没有Si原子
   "B",
]
rec_types = [
   "HD",
   "C",
   "A",
   "N",
   "NA",
   "OA",
   "F",
   "P",
   "SA",
   "S",
   "Cl",
   "Br",
   "I",
   "Mg",
   "Ca",
   "Mn",
   "Fe",
   "Zn",
]