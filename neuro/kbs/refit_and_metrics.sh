#!/bin/bash
set -e
cd ../../neuro
for s in Town05Data_IMU_Fusion Town03Data_IMU_Fusion Town07Data_IMU_Fusion \
         Town10HDData_IMU_Fusion; do
  d=data/${s}
  cp ${d}/slam_results_p2ft/trajectories.mat ${d}/slam_results/trajectories.mat
  cp ${d}/slam_results_p2ft/experiences.mat ${d}/slam_results/experiences.mat
  echo "回装 ${s}"
done
d=data/01_NeuroSLAM_Datasets/KITTI07Data_IMU_Fusion
cp ${d}/slam_results_p2ft/trajectories.mat ${d}/slam_results/trajectories.mat
cp ${d}/slam_results_p2ft/experiences.mat ${d}/slam_results/experiences.mat
echo "回装 KITTI07Data_IMU_Fusion"
$HOME/anaconda3/envs/tdmpc/bin/python kbs/compute_final_metrics.py
