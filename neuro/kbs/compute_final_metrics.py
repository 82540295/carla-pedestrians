#!/usr/bin/env python3
import numpy as np, pandas as pd, scipy.io as sio, json, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# NLM_DATA_ROOT=/path/to/datasets python3 compute_final_metrics.py
D = os.environ.get('NLM_DATA_ROOT', '../data/01_NeuroSLAM_Datasets').rstrip('/') + '/'
if not os.path.isdir(D):
    print(f'[ERROR] data root not found: {os.path.abspath(D)}\n'
             f'        set NLM_DATA_ROOT=<bundle>/datasets (or a full run tree) and retry')
    sys.exit(2)
OUT = {}

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
    return float(np.sqrt((err ** 2).mean())), float(err[-1] / L * 100)

def loadmat(d, var):
    m = sio.loadmat(D + d + '/slam_results/trajectories.mat')
    return np.array(m[var], dtype=float)

def loadcsv(d, fn, cols):
    p = D + d + '/' + fn
    if not os.path.exists(p):
        return None
    a = pd.read_csv(p)
    c = [x for x in cols if x in a.columns]
    return a[c].values if len(c) == 3 else None

t1 = {}
for d in ['Town05Data_IMU_Fusion', 'Town03Data_IMU_Fusion', 'Town07Data_IMU_Fusion',
          'Town10HDData_IMU_Fusion', 'KITTI07Data_IMU_Fusion']:
    base = d + '/'
    gt = loadcsv(d, 'ground_truth.txt', ['pos_x', 'pos_y', 'pos_z'])
    nlm = loadmat(d, 'exp_trajectory')
    fu = loadcsv(d, 'fusion_pose.txt', ['pos_x', 'pos_y', 'pos_z'])
    vo = loadcsv(d, 'visual_odometry.txt', ['vo_x', 'vo_y', 'vo_z'])
    if vo is None:
        pv = D.rstrip('/') + '/../KITTI_07/visual_odometry_ekf.txt'
        if os.path.exists(pv):
            a2 = pd.read_csv(pv)
            c2 = [x for x in ['vo_x', 'vo_y', 'vo_z'] if x in a2.columns]
            vo = a2[c2].values if len(c2) == 3 else None
    L = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    row = {'gt_len_m': round(L, 1), 'frames': int(min(len(gt), len(nlm)))}
    r, dr = sim3_full(nlm, gt); row['nlm'] = {'rmse': round(r, 2), 'drift': round(dr, 2)}
    if fu is not None and len(fu) > 50:
        r, dr = sim3_full(fu, gt); row['ekf'] = {'rmse': round(r, 2), 'drift': round(dr, 2)}
    else:
        row['ekf'] = None
    if vo is not None and len(vo) > 50:
        r, dr = sim3_full(vo, gt); row['vo'] = {'rmse': round(r, 2), 'drift': round(dr, 2)}
    else:
        row['vo'] = None
    p = D + base + 'slam_results/experiences.mat'
    if os.path.exists(p):
        m = sio.loadmat(p)
        NUM = int(float(m['NUM_EXPS'].ravel()[0]))
        EXP = m['EXPERIENCES']
        nvt = 0; nloop = 0
        n_exp = EXP.shape[1] if EXP.ndim == 2 else EXP.shape[0]
        for k in range(n_exp):
            e = EXP[0, k] if EXP.ndim == 2 else EXP[k]
            if e.shape != ():
                e = e.ravel()[0]
            try:
                vtid = int(float(np.atleast_1d(e['vt_id']).ravel()[0]))
                nvt += 1 if vtid > 0 else 0
            except Exception:
                pass
            try:
                nl = int(float(np.atleast_1d(e['numlinks']).ravel()[0]))
                if nl > 0:
                    lnks = e['links']
                    if lnks.shape != ():
                        lnks = np.atleast_1d(lnks)
                    for j in range(len(lnks)):
                        lid = lnks[j]
                        if lid.shape != ():
                            lid = lid.ravel()[0]
                        eid = int(float(np.atleast_1d(lid['exp_id']).ravel()[0]))
                        if 1 <= eid <= k:
                            nloop += 1
            except Exception:
                pass
        row['exp_nodes'] = NUM; row['templates'] = nvt; row['loop_edges'] = nloop
    t1[d] = row

OUT = {'table1': t1}
META = {
    'caliber': 'loop-line configuration (slam_results/), NOT the full system',
    'table3_source': 'se3_metrics.json -> NLM-fullsystem (slam_results_scale_nlm/)',
    'batch': '2026-10-14; closed-loop Town01 re-collected 2026-11',
}
with open('../kbs/final_metrics.json', 'w') as f:
    json.dump({'_meta': META, **OUT}, f, indent=2, ensure_ascii=False)

print('════ Table 1 (full Sim(3), Town05/03/07/10HD + KITTI07) ════')
print(f"{'set':24s} {'NLM':>7s} {'EKF':>7s} {'VO':>7s} | {'drift NLM/EKF':>14s} | nodes/tpl/loop | GTlen")
for d, r in t1.items():
    n = r['nlm']; e = r.get('ekf'); v = r.get('vo')
    ns = f"{n['rmse']:7.2f}"
    es = f"{e['rmse']:7.2f}" if e else '   ---'
    vs = f"{v['rmse']:7.2f}" if v else '   ---'
    ddr = f"{n['drift']}/{e['drift']}" if e else f"{n['drift']}/-"
    cnt = f"{r.get('exp_nodes','?')}/{r.get('templates','?')}/{r.get('loop_edges','?')}"
    print(f"{d:24s} {ns} {es} {vs} | {ddr:>14s} | {cnt:>14s} | {r['gt_len_m']}m")
print('\n→ 已写入 neuro/kbs/final_metrics.json')
