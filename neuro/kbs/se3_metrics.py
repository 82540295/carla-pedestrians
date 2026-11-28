#!/usr/bin/env python3
import numpy as np, pandas as pd, scipy.io as sio, json, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

D  = os.environ.get('NLM_DATA_ROOT', '../data/01_NeuroSLAM_Datasets').rstrip('/') + '/'
if not os.path.isdir(D):
    print(f'[ERROR] data root not found: {os.path.abspath(D)}\n'
          f'        set NLM_DATA_ROOT=<bundle>/datasets (or a full run tree) and retry')
    sys.exit(2)
ROOT_NOTE = '$NLM_DATA_ROOT/' if 'NLM_DATA_ROOT' in os.environ else D
BK = os.environ.get('NLM_BACKUP_ROOT', '../data/_p11_backup_20261014').rstrip('/') + '/'
SETS = ['Town05Data_IMU_Fusion', 'Town03Data_IMU_Fusion', 'Town07Data_IMU_Fusion',
        'Town10HDData_IMU_Fusion', 'KITTI07Data_IMU_Fusion', 'KITTI09Data_IMU_Fusion']
SHORT = {'Town05Data_IMU_Fusion': 'Town05', 'Town03Data_IMU_Fusion': 'Town03',
         'Town07Data_IMU_Fusion': 'Town07', 'Town10HDData_IMU_Fusion': 'Town10HD'}

def se3_rigid(est, gt):
    n = min(len(est), len(gt)); est, gt = est[:n], gt[:n]
    ec, gc = est - est.mean(0), gt - gt.mean(0)
    U, S, Vt = np.linalg.svd(gc.T @ ec)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    ea = est @ R + (gt.mean(0) - (est @ R).mean(0))
    err = np.linalg.norm(ea - gt, axis=1)
    L = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    return float(np.sqrt((err ** 2).mean())), float(err[-1] / L * 100)

def sim3_full(est, gt):
    n = min(len(est), len(gt)); est, gt = est[:n], gt[:n]
    ec, gc = est - est.mean(0), gt - gt.mean(0)
    U, S, Vt = np.linalg.svd(gc.T @ ec)
    d = np.linalg.det(Vt.T @ U.T)
    T = Vt.T @ np.diag([1, 1, d]) @ U.T
    s = (S @ np.array([1, 1, d])) / (ec ** 2).sum()
    ea = s * (est @ T) + (gt.mean(0) - s * (est.mean(0) @ T))
    err = np.linalg.norm(ea - gt, axis=1)
    L = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    return float(np.sqrt((err ** 2).mean())), float(err[-1] / L * 100), float(s)

def loadcsv(d, fn, cols):
    p = d + fn
    if not os.path.exists(p):
        return None
    a = pd.read_csv(p)
    c = [x for x in cols if x in a.columns]
    return a[c].values.astype(float) if len(c) == 3 else None

def loadmat(d, fn, var):
    p = d + fn
    if not os.path.exists(p):
        return None
    try:
        return np.array(sio.loadmat(p)[var], dtype=float)
    except Exception:
        return None

OUT = {}
for sd in SETS:
    base = D + sd + '/'
    gt = loadcsv(base, 'ground_truth.txt', ['pos_x', 'pos_y', 'pos_z'])
    if gt is None:
        print(f'MISSING GT: {sd}')
        continue
    cfg = {
        'VO': loadcsv(base, 'visual_odometry.txt', ['vo_x', 'vo_y', 'vo_z']),
        'EKF': loadcsv(base, 'fusion_pose.txt', ['pos_x', 'pos_y', 'pos_z']),
        'EKF+scale': None,
        'NLM-loopline': loadmat(base, 'slam_results/trajectories.mat', 'exp_trajectory'),
        'NLM-fullsystem': loadmat(base, 'slam_results_scale_nlm/trajectories.mat', 'exp_trajectory'),
        'ORB-only': loadmat(base, 'slam_results_orbonly/trajectories.mat', 'exp_trajectory'),
    }
    if cfg['VO'] is None:
        cfg['VO'] = loadcsv(D + '../KITTI_07/', 'visual_odometry_ekf.txt', ['vo_x', 'vo_y', 'vo_z'])
    if sd in SHORT:
        cfg['EKF+scale'] = loadcsv(BK + SHORT[sd] + '/', 'fp_head030.txt', ['pos_x', 'pos_y', 'pos_z'])
    row = {'gt_len_m': round(float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()), 1)}
    for name, traj in cfg.items():
        if traj is None or len(traj) < 50:
            continue
        r_s, dr_s = se3_rigid(traj, gt)
        r_f, dr_f, s_f = sim3_full(traj, gt)
        row[name] = {'se3_rmse': round(r_s, 2), 'se3_drift': round(dr_s, 2),
                     'sim3_rmse': round(r_f, 2), 'sim3_drift': round(dr_f, 2),
                     'sim3_fitted_scale': round(s_f, 4)}
    OUT[sd] = row

PREV = {}
JSN = '../kbs/se3_metrics.json'
if os.path.exists(JSN):
    try:
        PREV = json.load(open(JSN))
    except Exception:
        PREV = {}
SNAP = {k: v for k, v in PREV.items() if not k.startswith('_')}
for sd, prow in SNAP.items():
    row = OUT.get(sd)
    if row is None:
        carried = {k: (dict(v, carried=True) if isinstance(v, dict) else v)
                   for k, v in prow.items() if k != 'gt_len_m'}
        OUT[sd] = {'gt_len_m': prow.get('gt_len_m'), **carried, 'carried_row': True}
        continue
    for cfg, v in prow.items():
        if cfg == 'gt_len_m' or not isinstance(v, dict):
            continue
        if cfg not in row:
            row[cfg] = dict(v, carried=True)

OUT['_meta'] = {
    'calibers': {
        'sim3_rmse': 'full Sim(3) refit (shape-only upper bound)',
        'se3_rmse': 'SE(3) rigid Kabsch, no scale (metric-scale caliber)',
        'drift': 'terminal error / trajectory length (%)',
    },
    'data_root': ROOT_NOTE,
    'data_root_note': 'token for the directory that was evaluated: the released bundle'
                      ' dataset root (NLM_Repro_Assets_*/datasets/) in place of $NLM_DATA_ROOT,'
                      ' or the repo default ../data/01_NeuroSLAM_Datasets/',
    'note': 'Keys computed from the data root are plain. Keys with "carried": true were '
            'computed in an earlier batch and their artifact is NOT under the data root '
            '(EKF+scale needs _p11_backup_20261014/<seq>/fp_head030.txt, NLM-fullsystem needs '
            'slam_results_scale_nlm/, ORB-only needs slam_results_orbonly/); they are '
            'carried over from the previous JSON so that Table 4 stays complete, and are '
            'replaced the moment the artifact is present.',
}
present, carried = set(), set()
for sd, row in OUT.items():
    if sd.startswith('_') or not isinstance(row, dict):
        continue
    for k, v in row.items():
        if k in ('gt_len_m', 'carried_row') or not isinstance(v, dict):
            continue
        (carried if v.get('carried') or row.get('carried_row') else present).add(f'{sd}/{k}')
OUT['_meta']['configs_computed'] = sorted(present)
OUT['_meta']['configs_carried'] = sorted(carried)
OUT['_meta']['table3_notes'] = [
    'KITTI07 Ours cell (18.45 / 29.99) equals NLM-loopline: the scale branch is inactive '
    'zero-shot on real imagery (Supplementary S.1), so the full system there IS the loop line.',
    'Town10HD Ours == EKF+scale (14.44): the single accepted loop link is inert on the open layout.',
    'Table 4 EKF+scale and ORB-only cells come from artifacts outside the released asset bundle '
    'unless export_repro_bundle.py has been run over the original run tree.',
]

N_COMPUTED = sum(1 for sd, row in OUT.items() if not sd.startswith('_')
                 for k, v in row.items()
                 if k not in ('gt_len_m', 'carried_row') and isinstance(v, dict)
                 and not v.get('carried') and not row.get('carried_row'))
if N_COMPUTED == 0:
    print(f'[ERROR] nothing was recomputed under {os.path.abspath(D)} -- refusing to write se3_metrics.json '
          f'and refusing to report a cross-check\n'
          f'        set NLM_DATA_ROOT=<bundle>/datasets (or a full run tree) and retry')
    sys.exit(2)

with open('../kbs/se3_metrics.json', 'w') as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False)
    f.write('\n')

CMPS = ['VO', 'EKF', 'EKF+scale', 'NLM-loopline', 'NLM-fullsystem', 'ORB-only']
print('=' * 96)
for sd, row in OUT.items():
    if sd.startswith('_'):
        continue
    print(f"\n### {sd}  (GT {row['gt_len_m']}m)")
    print(f"{'配置':12s} {'SE(3)刚性':>10s} {'drift%':>8s} | {'sim3_full':>10s} {'drift%':>8s} {'拟合s':>9s}")
    for c in CMPS:
        if c in row:
            m = row[c]
            print(f"{c:12s} {m['se3_rmse']:10.2f} {m['se3_drift']:8.2f} | "
                  f"{m['sim3_rmse']:10.2f} {m['sim3_drift']:8.2f} {m['sim3_fitted_scale']:9.4f}")
print('\n→ 已写入 neuro/kbs/se3_metrics.json')

FM = json.load(open('../kbs/final_metrics.json'))
t1 = FM['table1']
print('\n── 与 final_metrics.json 交叉校验 (sim3_full 应一致) ──')
ok = True
NCMP = 0
for sd, row in OUT.items():
    if sd.startswith('_') or sd not in t1:
        continue
    for name, key in [('NLM-loopline', 'nlm'), ('EKF', 'ekf'), ('VO', 'vo')]:
        ref = t1[sd].get(key)
        if name in row and ref and not row[name].get('carried'):
            a, b = row[name]['sim3_rmse'], ref['rmse']
            NCMP += 1
            mark = 'OK' if abs(a - b) < 0.011 else 'MISMATCH'
            if mark != 'OK':
                ok = False
            print(f'  {sd:28s} {name:10s} {a:8.2f} vs {b:8.2f}  {mark}')
print('CROSS-CHECK', 'PASS' if ok else 'FAIL', f'({NCMP} computed cells compared)')
sys.exit(0 if ok else 1)
