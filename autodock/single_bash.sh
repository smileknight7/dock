#!/bin/bash                                                                                                                                                               
# 单样本 AutoDock4 对接，等价于 docking.py 的 single 模式                                                                                                                 
set -euo pipefail                                                                                                                                                         

AUTODOCK4=/data3/lpw/software/autodock4/x86_64Linux2/autodock4                                                                                                            
                
DPF=${1:-/data3/lpw/dock_workdir/autodock/test/ligand_0000_3I03_protein_pocket.dpf}                                                                                       
OUTPUT_DIR=${2:-/data3/lpw/dock_workdir/autodock/test}
                                                                                                                                                                        
[[ -f "$DPF" ]] || { echo "DPF not found: $DPF" >&2; exit 1; }                                                                                                            
                                                                                                                                                                        
mkdir -p "$OUTPUT_DIR"                                                                                                                                                    
                
DPF_DIR=$(dirname "$DPF")                                                                                                                                                 
DPF_NAME=$(basename "$DPF")
DLG="$OUTPUT_DIR/${DPF_NAME%.dpf}.dlg"                                                                                                                                    
                                                                                                                                                                        
echo "DPF: $DPF"
echo "DLG: $DLG"                                                                                                                                                          
echo "cwd: $DPF_DIR"
echo "----"                                                                                                                                                               

# AutoDock4 必须在 DPF 所在目录跑（DPF 里引用的 .map / .pdbqt 是相对路径）                                                                                                
cd "$DPF_DIR"   
"$AUTODOCK4" -p "$DPF_NAME" -l "$DLG"                                                                                                                                     
                                                                                                                                                                        
echo "----"
echo "Top binding energies:"                                                                                                                                              
grep "Estimated Free Energy of Binding" "$DLG" | head -5 