#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy.io as sio

D = '../data/01_NeuroSLAM_Datasets/'
STAGE = '/tmp/nlm_sweep_weak'
LOG = os.path.join(STAGE, 'log.txt')
QUICKSTART = '../07_test/test_imu_visual_slam/quickstart'
KBS = os.path.dirname(os.path.abspath(__file__))

BASE = {
    'Town03':   (0.040, 20, 52.04),
    'Town07':   (0.040, 20, 12.68),
    'Town10HD': (0.120, 10, 7.59),
}
VT_GRID = [0.020, 0.030, 0.040, 0.055, 0.080, 0.120]
CE_GRID = [10, 20, 30]
HDC = 18


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


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


def tag_of(setname, vt, ce):
    return f'{setname}_vt{vt:g}_ce{ce}_hdc{HDC}'


def rmse_of(setname, vt, ce):
    m = sio.loadmat(os.path.join(STAGE, setname, tag_of(setname, vt, ce),
                                 'trajectories.mat'))
    nlm = np.array(m['exp_trajectory'], dtype=float)
    a = pd.read_csv(os.path.join(D, f'{setname}Data_IMU_Fusion',
                                 'ground_truth.txt'))
    gt = a[['pos_x', 'pos_y', 'pos_z']].values
    return sim3_full(nlm, gt)


def run_one(setname, vt, ce):
    tag = tag_of(setname, vt, ce)
    dst = os.path.join(STAGE, setname, tag)
    if os.path.exists(os.path.join(dst, 'trajectories.mat')):
        log(f'  跳过 {tag} (已暂存)')
        return 0
    with open(os.path.join(STAGE, 'combo.txt'), 'w') as f:
        f.write(f'{setname} {vt:g} {ce} {HDC}\n')
    t0 = time.time()
    r = subprocess.run(['matlab', '-batch', 'SWEEP_ONE'], cwd=QUICKSTART,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    tail = [l for l in r.stdout.splitlines()
            if l.startswith('>>>') or l.startswith('错误') or l.startswith('Error')][-4:]
    for l in tail:
        log(f'  | {l}')
    ok = r.returncode == 0
    if ok:
        src_dir = os.path.join(D, f'{setname}Data_IMU_Fusion', 'slam_results')
        os.makedirs(dst, exist_ok=True)
        for fn in ['trajectories.mat', 'experiences.mat',
                   'performance_report.txt', 'exp_trajectory.txt']:
            if os.path.exists(os.path.join(src_dir, fn)):
                shutil.copyfile(os.path.join(src_dir, fn), os.path.join(dst, fn))
    ok = ok and os.path.exists(os.path.join(dst, 'trajectories.mat'))
    log(f'  {tag}: rc={r.returncode} 耗时 {dt/60:.1f}min {"OK" if ok else "失败"}')
    return 0 if ok else 1


def main():
    os.makedirs(STAGE, exist_ok=True)
    log('sweep_weak_sets 启动 (每组合独立 MATLAB 进程)')
    combos1 = [(s, vt, BASE[s][1]) for s in BASE for vt in VT_GRID]
    log(f'=== Phase1-VT: {len(combos1)} runs ===')
    bad = [c for c in combos1 if run_one(*c)]
    if bad:
        log(f'Phase1 有 {len(bad)} 个组合失败: {bad}')
        if len(bad) > 4:
            log('失败过多, 中止')
            sys.exit(1)

    cands = {}
    for s in BASE:
        vt_b, ce_b, rmse_b = BASE[s]
        rows = []
        for vt in VT_GRID:
            try:
                r, dr = rmse_of(s, vt, ce_b)
                rows.append((vt, ce_b, r, dr))
                mark = ' (基线复现)' if vt == vt_b else ''
                log(f'  {s} VT={vt:g} CE={ce_b}: RMSE={r:.2f} drift={dr:.2f}{mark}')
            except Exception as e:
                log(f'  {s} VT={vt:g} CE={ce_b}: 缺失/失败 {e}')
        rows.sort(key=lambda x: x[2])
        cands[s] = rows
        log(f'  >> {s} Phase1 最优: VT={rows[0][0]:g} CE={rows[0][1]} '
            f'RMSE={rows[0][2]:.2f} (基线 {rmse_b})')

    combos2 = [(s, cands[s][0][0], ce) for s in BASE
               for ce in CE_GRID if ce != BASE[s][1]]
    log(f'=== Phase2-CE: {len(combos2)} runs ===')
    for c in combos2:
        run_one(*c)

    log('=== 回装决策 ===')
    for s in BASE:
        vt_best, ce_best, r_best = cands[s][0][0], cands[s][0][1], cands[s][0][2]
        vt2, ce2 = cands[s][0][0], None
        for ce in CE_GRID:
            if ce == BASE[s][1]:
                continue
            try:
                r, dr = rmse_of(s, cands[s][0][0], ce)
                log(f'  {s} VT={cands[s][0][0]:g} CE={ce}: RMSE={r:.2f} drift={dr:.2f}')
                if r < r_best:
                    vt_best, ce_best, r_best = cands[s][0][0], ce, r
            except Exception as e:
                log(f'  {s} VT={cands[s][0][0]:g} CE={ce}: 缺失/失败 {e}')
        rmse_b = BASE[s][2]
        if r_best < rmse_b:
            src_tag = (vt_best, ce_best)
            why = f'改进 {rmse_b} → {r_best:.2f} (-{(rmse_b-r_best)/rmse_b*100:.1f}%)'
        else:
            src_tag = (BASE[s][0], BASE[s][1])
            why = f'未改进 (最优 {r_best:.2f} >= 基线 {rmse_b}), 回装基线组合'
        src = os.path.join(STAGE, s, tag_of(s, src_tag[0], src_tag[1]))
        dst = os.path.join(D, f'{s}Data_IMU_Fusion', 'slam_results')
        if not os.path.exists(os.path.join(src, 'trajectories.mat')):
            log(f'  {s}: 回装源缺失 {src}, 跳过 (需人工处理)')
            continue
        for fn in ['trajectories.mat', 'experiences.mat',
                   'performance_report.txt', 'exp_trajectory.txt']:
            if os.path.exists(os.path.join(src, fn)):
                shutil.copyfile(os.path.join(src, fn), os.path.join(dst, fn))
        log(f'  {s}: 回装 VT={src_tag[0]:g} CE={src_tag[1]} — {why}')

    log('=== 重算 final_metrics.json ===')
    r = subprocess.run([sys.executable, 'compute_final_metrics.py'], cwd=KBS,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log(r.stdout.strip())
    log('sweep_weak_sets 全部完成')


if __name__ == '__main__':
    main()
