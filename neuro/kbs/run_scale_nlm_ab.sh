#!/bin/bash
set -e
cd ../../neuro
DATA=data
BK=data/_p11_backup_20261014
M=$HOME/下载/MATLAB/matlab_2021b_linux_clean/matlab_2021b/bin/matlab
QS=07_test/test_imu_visual_slam/quickstart
for s in Town03 Town05 Town07 Town10HD; do
  d=${DATA}/${s}Data_IMU_Fusion
  echo "########## ${s} ##########"
  cp ${d}/fusion_pose.txt ${BK}/${s}/fp_live_nlm.txt
  cp ${BK}/${s}/fp_head030.txt ${d}/fusion_pose.txt
  (cd ${QS} && ${M} -batch "P2S_${s}" 2>&1 | grep -E "A-fix|完成" | tail -2)
  cp ${BK}/${s}/fp_live_nlm.txt ${d}/fusion_pose.txt
done
echo "SCALE_NLM_AB_DONE"
