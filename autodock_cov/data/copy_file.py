import shutil
from pathlib import Path


pose_dir = Path('/data3/lpw/dock_workdir/autodock_cov/pose4')
result_dir = Path('/data3/lpw/dock_workdir/autodock_cov/result4')
preparation_dir = Path('/data3/lpw/dock_workdir/autodock_cov/preparation4')             # 这里后面修改的时候要复制一下gt的pose文件方便看对接盒子

pdb_ids = [
    "2TOD",
    "1ZPZ",
    "4XZ0",
    "5GNK",
    "1F2C",
    "5UG9",
    "4I24",
    "1YK7",
    "3IKA",
    "4PDZ",
    "2ZK4",
]

analysis_dir = Path('/data3/lpw/dock_workdir/autodock_cov/analysis_large_rmsd')
analysis_dir.mkdir(exist_ok=True, parents=True)
for pdb_id in pdb_ids:
    pose_subdir = pose_dir / pdb_id
    result_subdir = result_dir / pdb_id
    preparation_subdir = preparation_dir / pdb_id
    if pose_subdir.exists() and result_subdir.exists():
        shutil.copytree(pose_subdir, analysis_dir / 'pose' / pdb_id, dirs_exist_ok=True)
        shutil.copytree(result_subdir, analysis_dir / 'result' / pdb_id, dirs_exist_ok=True)
        shutil.copytree(preparation_subdir, analysis_dir / 'preparation' / pdb_id, dirs_exist_ok=True)
    else:
        print(f"跳过 {pdb_id}:目录不存在")