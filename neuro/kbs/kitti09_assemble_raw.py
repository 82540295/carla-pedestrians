#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    <unzip_root>/2011_09_30/2011_09_30_drive_0033_sync/image_00/{data/*.png, timestamps.txt}
    <unzip_root>/2011_09_30/2011_09_30_drive_0033_sync/oxts/{data/*.txt, timestamps.txt}
"""
import os
import sys
import shutil
from datetime import datetime

UNZIP_SUB = '2011_09_30/2011_09_30_drive_0033_sync'


def parse_ts_line(line):
    line = line.strip()
    if not line:
        return None
    main, frac = line, 0.0
    if '.' in line:
        main, f = line.split('.', 1)
        frac = int((f + '000000000')[:9]) / 1e9
    return datetime.strptime(main, '%Y-%m-%d %H:%M:%S'), frac


def load_ts(path):
    rows = []
    for line in open(path):
        p = parse_ts_line(line)
        if p is not None:
            rows.append(p)
    return rows


def to_rel(rows):
    d0, f0 = rows[0]
    return [(d - d0).total_seconds() + (f - f0) for d, f in rows]


def main():
    unzip_root, raw = sys.argv[1], sys.argv[2]
    src = os.path.join(unzip_root, UNZIP_SUB)
    assert os.path.isdir(src), f'未找到解压目录: {src}'

    pose_path = os.path.join(raw, 'poses.txt')
    assert os.path.exists(pose_path), f'{pose_path} 缺失 (先从 odometry zip 抽取)'
    n_pose = sum(1 for _ in open(pose_path))

    img_dir = os.path.join(src, 'image_00', 'data')
    imgs = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
    ts_rows = load_ts(os.path.join(src, 'image_00', 'timestamps.txt'))
    print(f'[IMG] raw 驱动帧数={len(imgs)}  timestamps={len(ts_rows)}  poses={n_pose}')
    assert len(ts_rows) == len(imgs), '图像数与时间戳数不一致'
    n_img = len(imgs)
    if n_img > n_pose:
        print(f'[TRUNC] raw {n_img} 帧 → 截取前 {n_pose} 帧 (odometry 09 官方映射从帧 0 起)')
        imgs = imgs[:n_pose]
        ts_rows = ts_rows[:n_pose]
        n_img = n_pose
    dst_img = os.path.join(raw, 'image_0')
    os.makedirs(dst_img, exist_ok=True)
    for i, f in enumerate(imgs):
        shutil.copy2(os.path.join(img_dir, f), os.path.join(dst_img, f'{i:06d}.png'))

    ts_rel = to_rel(ts_rows)
    with open(os.path.join(raw, 'times.txt'), 'w') as f:
        for t in ts_rel:
            f.write(f'{t:.6e}\n')
    dt = [ts_rel[i + 1] - ts_rel[i] for i in range(n_img - 1)]
    print(f'[TIME] times.txt: {n_img} 行  dt_mean={sum(dt)/len(dt):.4f}s  '
          f'dt_range=[{min(dt):.4f},{max(dt):.4f}]')

    oxts_data = os.path.join(src, 'oxts', 'data')
    o_files = sorted(f for f in os.listdir(oxts_data) if f.endswith('.txt'))
    dst_oxts = os.path.join(raw, 'oxts', 'data')
    os.makedirs(dst_oxts, exist_ok=True)
    if len(o_files) >= n_img:
        for i in range(n_img):
            shutil.copy2(os.path.join(oxts_data, o_files[i]),
                         os.path.join(dst_oxts, f'{i:010d}.txt'))
        print(f'[OXTS] 1:1 按帧号对齐 {n_img} 文件 (源 {len(o_files)}, 截尾 {len(o_files)-n_img})')
    else:
        raise AssertionError(f'oxts({len(o_files)}) < 帧数({n_img}), 无法 1:1 对齐')

    assert os.path.exists(os.path.join(raw, 'calib.txt')), 'calib.txt 缺失'
    assert n_img == n_pose, f'帧数 {n_img} != poses {n_pose}'
    print(f'[OK] raw 布局就绪: {raw}  (帧数={n_img})')


if __name__ == '__main__':
    main()
