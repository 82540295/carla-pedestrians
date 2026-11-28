#!/usr/bin/env python3
import os
import argparse

import numpy as np
import torch
import torch.nn as nn

D = '../data/01_NeuroSLAM_Datasets'
TRAIN_SETS = ['Town05Data_IMU_Fusion', 'Town03Data_IMU_Fusion', 'Town07Data_IMU_Fusion']
OUT = '../kbs/pr_head.pt'


def load_set(name):
    sub = os.path.join(D, name)
    emb = np.load(os.path.join(sub, 'slam_results', 'pr_dinov2s.npy')).astype(np.float32)
    lab = np.load(os.path.join(sub, 'slam_results', 'pr_loop_labels.npz'))
    return emb, lab['place_id'], lab['hard_neg']


class Head(nn.Module):
    def __init__(self, din=384, dmid=384, dout=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, dmid), nn.LayerNorm(dmid), nn.GELU(),
            nn.Linear(dmid, dout),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--margin', type=float, default=0.3)
    ap.add_argument('--neg-m', type=float, default=0.3)
    ap.add_argument('--pos-floor', type=float, default=0.7)
    ap.add_argument('--neg-ceiling', type=float, default=-0.3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--pos-cap', type=int, default=25,
                    help='每场所每 epoch 正对上限 (2026-10-14: 3 巨型密集簇各抽 85 正对'
                         '把相似度空间压饱和 — Town05/07/10HD τ=0.9 各 ~200 假事件; '
                         '上限 25 后正负信号再平衡)')
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    embs, pids, hnegs = [], [], []
    for name in TRAIN_SETS:
        e, pid, hn = load_set(name)
        embs.append(e); pids.append(pid); hnegs.append(hn)
    FEAT = torch.from_numpy(np.concatenate(embs, axis=0)).cuda()      # (Nt,384)
    off = np.concatenate([[0], np.cumsum([len(x) for x in embs[:-1]])])
    PID = np.concatenate(pids)
    HN = np.concatenate([hn + o for hn, o in zip(hnegs, off)])

    revisit_idx = np.where(PID >= 0)[0]
    groups = {}
    for i in revisit_idx:
        groups.setdefault(int(PID[i]), []).append(i)
    group_members = {k: torch.from_numpy(np.array(v, dtype=np.int64)).cuda()
                     for k, v in groups.items() if len(v) >= 2}
    group_keys = list(group_members.keys())
    hasneg_idx = np.where(HN[:, 0] >= 0)[0]
    print(f"训练帧={FEAT.shape[0]} 场所={len(group_keys)} "
          f"(成员={{{', '.join(str(k) + ':' + str(group_members[k].shape[0]) for k in group_keys)}}}) "
          f"有难负帧={len(hasneg_idx)}", flush=True)

    model = Head().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        model.train()
        anchors, positives = [], []
        per = min(max(8, args.batch // max(1, len(group_keys) * 2)), args.pos_cap)
        for gk in group_keys:
            mem = group_members[gk]
            n = mem.shape[0]
            t = min(n, per)
            a = mem[torch.randperm(n)[:t]]
            p = mem[torch.randperm(n)[:t]]
            same = (a == p)
            if same.any():
                p[same] = (a[same] + 1) % n
            anchors.append(a)
            positives.append(p)
        a = torch.cat(anchors)
        p = torch.cat(positives)
        hn = torch.from_numpy(HN[a.cpu().numpy(), 0]).cuda()
        ok = hn >= 0
        a, p, hn = a[ok], p[ok], hn[ok]

        ea = model(FEAT[a])
        ep_ = model(FEAT[p])
        en = model(FEAT[hn])
        s_ap = (ea * ep_).sum(1)
        s_an = (ea * en).sum(1)
        tri = torch.clamp(args.pos_floor - s_ap, min=0.0).mean() \
            + torch.clamp(s_an - args.neg_ceiling, min=0.0).mean()

        j = torch.from_numpy(np.random.choice(
            hasneg_idx, size=args.batch,
            replace=len(hasneg_idx) < args.batch)).cuda()
        jhn = torch.from_numpy(HN[j.cpu().numpy(), 0]).cuda()
        okj = jhn >= 0
        j, jhn = j[okj], jhn[okj]
        ej = model(FEAT[j])
        ejn = model(FEAT[jhn])
        hinge = torch.clamp((ej * ejn).sum(1) + args.neg_m, min=0.0).mean()

        loss = tri + hinge
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep{ep:3d}  loss={loss.item():.4f}  tri={tri.item():.4f} "
                  f"hinge={hinge.item():.4f}  anchors={a.shape[0]}", flush=True)

    torch.save(dict(state=model.state_dict(), dout=128, din=384,
                    train_sets=TRAIN_SETS, margin=args.margin, neg_m=args.neg_m), OUT)
    print(f"saved {OUT}  final_loss={loss.item():.4f}", flush=True)


if __name__ == '__main__':
    main()
