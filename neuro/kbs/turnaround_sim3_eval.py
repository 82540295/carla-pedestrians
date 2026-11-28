#!/usr/bin/env python3
"""
trajectories.mat (exp_traj_aligned / fusion_pos_aligned / pure_visual_aligned /
"""
import os

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.environ.get('NLM_DATA_ROOT',
                                   os.path.join(HERE, '..', 'data')), 'Town01Turnaround_IMU_Fusion')


def align_metric(traj, gt, anchor=100):
    n = min(len(traj), len(gt)); traj, gt = traj[:n], gt[:n]
    a_gt, a_tr = gt[:anchor], traj[:anchor]
    gs, es = a_gt - a_gt.mean(0), a_tr - a_tr.mean(0)
    U, S, Vt = np.linalg.svd(gs.T @ es)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = S.sum() / max(np.sum(es ** 2), 1e-12)
    t = a_gt.mean(0) - (s * R) @ a_tr.mean(0)
    aligned = (s * (R @ traj.T)).T + t
    err = np.linalg.norm(aligned - gt, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    final = float(err[-1])
    L = float(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1).sum())
    drift = 100.0 * final / L if L > 0 else float('nan')
    return rmse, final, drift, L


def main():
    print('=== 折返集(10000帧) 论文口径 Sim(3) 对齐 ATE ===')
    for label, sub, traj_var in [
        ('NLM(dc基线)', 'dc_final', 'exp_traj_aligned'),
        ('NLM(+闭环约束)', 'a_final', 'exp_traj_aligned'),
        ('EKF', 'a_final', 'fusion_pos_aligned'),
        ('VO', 'a_final', 'pure_visual_aligned'),
    ]:
        mat = loadmat(os.path.join(BASE, sub, 'trajectories.mat'))
        traj = np.asarray(mat[traj_var], dtype=float)
        gt = np.asarray(mat['gt_pos_aligned'], dtype=float)
        rmse, final, drift, L = align_metric(traj, gt)
        print(f'{label:14s} RMSE={rmse:8.2f}m  final={final:8.2f}m  '
              f'drift={drift:6.2f}%  (gt_len={L:.0f}m)')


if __name__ == '__main__':
    main()
