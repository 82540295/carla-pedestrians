#!/usr/bin/env python3
import os
import sys

import numpy as np

D = '../data/01_NeuroSLAM_Datasets'
SETS = ['Town05Data_IMU_Fusion', 'Town03Data_IMU_Fusion', 'Town07Data_IMU_Fusion',
        'Town10HDData_IMU_Fusion', 'KITTI07Data_IMU_Fusion']

R_M = 10.0
ARC_MIN = 50.0
ARC_MAX_NEG = 1500.0
K_NEG = 4


def load_gt(sub):
    with open(os.path.join(sub, 'ground_truth.txt')) as f:
        f.readline()
        a = np.loadtxt(f, delimiter=',', usecols=(1, 2, 3))
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))])
    return a, arc


def mine(sub, k_neg=K_NEG):
    p, arc = load_gt(sub)
    N = len(p)
    emb = np.load(os.path.join(sub, 'slam_results', 'pr_dinov2s.npy')).astype(np.float32)

    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    BLK = 500
    n_edges = 0
    for s in range(0, N, BLK):
        ii = np.arange(s, min(s + BLK, N))
        for j0 in range(0, s, BLK):
            jj = np.arange(j0, min(j0 + BLK, s))
            d = np.linalg.norm(p[ii, None, :] - p[None, jj, :], axis=2)
            da = arc[ii][:, None] - arc[jj][None, :]
            m = (d < R_M) & (da > ARC_MIN) & (da < 3000)
            for r, c in zip(*np.nonzero(m)):
                union(int(ii[r]), int(jj[c]))
                n_edges += 1

    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    place_id = np.full(N, -1, dtype=np.int32)
    lab = 0
    n_pos_pairs = 0
    for g in groups.values():
        if len(g) >= 2:
            for i in g:
                place_id[i] = lab
            n_pos_pairs += len(g) * (len(g) - 1) // 2
            lab += 1

    S = emb @ emb.T
    hard_neg = np.full((N, k_neg), -1, dtype=np.int32)
    for i in range(N):
        lo = np.searchsorted(arc, arc[i] - ARC_MAX_NEG, side='right')
        hi = np.searchsorted(arc, arc[i] - ARC_MIN, side='right')
        if hi <= lo:
            continue
        cand = np.arange(lo, hi)
        d = np.linalg.norm(p[i] - p[cand], axis=1)
        cand = cand[d > R_M]
        if len(cand) == 0:
            continue
        sims = S[i, cand]
        top = np.argsort(-sims)[:k_neg]
        hard_neg[i] = cand[top]

    out = os.path.join(sub, 'slam_results', 'pr_loop_labels.npz')
    np.savez(out, place_id=place_id, hard_neg=hard_neg)
    n_revisit = int((place_id >= 0).sum())
    print(f"[{os.path.basename(sub)}] N={N} 重访帧={n_revisit} 场所={lab} "
          f"重访边={n_edges} 正对={n_pos_pairs} 难负帧={int((hard_neg[:, 0] >= 0).sum())}",
          flush=True)
    return dict(N=N, n_revisit=n_revisit, n_places=lab, n_edges=n_edges,
                n_pos_pairs=n_pos_pairs, n_hard=int((hard_neg[:, 0] >= 0).sum()))


if __name__ == '__main__':
    only = sys.argv[1:] or None
    for name in SETS:
        if only and name not in only:
            continue
        mine(os.path.join(D, name))
    print('DONE', flush=True)
