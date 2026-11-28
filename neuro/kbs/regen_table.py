#!/usr/bin/env python3
"""
  [_, est_a, T] = procrustes(gt, est, 'Scaling', true);  est_a = T.b*est + T.c
"""
import numpy as np, os, os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('NLM_DATA_ROOT', os.path.join(_HERE, '..', 'data', '01_NeuroSLAM_Datasets')).rstrip('/')

def load_pos(path, poscols=None):
    with open(path) as f:
        first = f.readline().strip()
    try:
        [float(x) for x in first.split(',')]
        header = False
    except ValueError:
        header = True
    a = np.loadtxt(path, delimiter=',', skiprows=1 if header else 0)
    cols = poscols if poscols is not None else (1, 2, 3) if header else (0, 1, 2)
    return a[:, list(cols)]

def procrustes_7dof(gt, est, anchor=None):
    n = min(len(gt), len(est)); gt = gt[:n]; est = est[:n]
    a = n if anchor is None else min(anchor, n)
    gs = gt[:a] - gt[:a].mean(0); es = est[:a] - est[:a].mean(0)
    H = gs.T @ es
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = S.sum() / max(np.sum(es**2), 1e-12)
    t = gt[:a].mean(0) - (s * R) @ est[:a].mean(0)
    est_a = (s * (R @ est.T)).T + t
    return np.linalg.norm(est_a - gt, axis=1), s

def metrics(gt, est, anchor=None):
    d, s = procrustes_7dof(gt, est, anchor)
    rmse = float(np.sqrt((d**2).mean()))
    term = float(d[-1])
    return rmse, term, s

gt01 = load_pos(f'{ROOT}/Town01Data_IMU_Fusion/ground_truth.txt')
ekf01 = load_pos(f'{ROOT}/Town01Data_IMU_Fusion/fusion_pose.txt')
vo01 = load_pos(f'{ROOT}/Town01Data_IMU_Fusion/visual_odometry.txt')
nlm01_old = load_pos(f'{ROOT}/Town01Data_IMU_Fusion/slam_results_backup_1003/exp_trajectory.txt')
nlm01_new = load_pos(f'{ROOT}/Town01Data_IMU_Fusion/slam_results/exp_trajectory.txt')

print('=== 口径验证 (论文: NLM=16.63 EKF=15.94 VO=36.46) ===')
for label, est in [('NLM_10-03', nlm01_old), ('NLM_new(P1ON)', nlm01_new)]:
    for an, aname in [(None, '全轨迹'), (100, '前100帧')]:
        r, t, s = metrics(gt01, est, an)
        print(f'  {label:<14} {aname}: RMSE={r:6.2f}  终点={t:6.2f}  scale={s:.3f}')
for label, est in [('EKF', ekf01), ('VO', vo01)]:
    for an, aname in [(None, '全轨迹'), (100, '前100帧')]:
        r, t, s = metrics(gt01, est, an)
        print(f'  {label:<14} {aname}: RMSE={r:6.2f}  终点={t:6.2f}  scale={s:.3f}')
