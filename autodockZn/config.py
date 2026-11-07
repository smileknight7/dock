#!/usr/bin/env python3
"""Shared configuration for AutoDock-GPU helper scripts."""


from pathlib import Path


ADGPU_BINARY = Path("/data3/lpw/software/autodockgpu/adgpu-v1.6_linux_x64_cuda11_128wi")
AUTOGRID4_BINARY = Path("/data3/lpw/software/autodock4/ADFRsuite-1.0/bin/autogrid4")
CONDA_PREFIX = Path("/data3/lpw/conda/envs/autodock-gpu")



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
   # "B",  # AD4Zn.dat has no B atom_par; add validated parameters before enabling.
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
