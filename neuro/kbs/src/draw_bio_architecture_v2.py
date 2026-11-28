#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Polygon
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'fig')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(1, 1, figsize=(22, 7))
ax.set_xlim(0, 22)
ax.set_ylim(0, 7)
ax.axis('off')

# ==========================
# ==========================
COLOR_CARLA = '#2E86AB'
COLOR_VESTIBULAR = '#FF6B35'
COLOR_VISUAL = '#06A77D'
COLOR_VENTRAL = '#F77F00'
COLOR_DORSAL = '#9B59B6'
COLOR_FUSION = '#F4D35E'
COLOR_ATTENTION = '#EE6C4D'
COLOR_DECISION = '#3D5A80'

# ==========================
# ==========================
carla_box = FancyBboxPatch((0.5, 1.5), 2.5, 4, 
                           boxstyle="round,pad=0.15", 
                           facecolor=COLOR_CARLA, 
                           edgecolor='white', 
                           linewidth=3, 
                           alpha=0.85)
ax.add_patch(carla_box)

road1 = Polygon([(1, 4.5), (2.5, 4.5), (2.3, 3.8), (1.2, 3.8)], 
                facecolor='white', alpha=0.3)
road2 = Polygon([(1, 2.8), (2.5, 2.8), (2.3, 2.1), (1.2, 2.1)], 
                facecolor='white', alpha=0.3)
ax.add_patch(road1)
ax.add_patch(road2)

ax.text(1.75, 5.3, 'CARLA', ha='center', fontsize=14, weight='bold', color='white')
ax.text(1.75, 4.2, 'Town01', ha='center', fontsize=11, color='white', style='italic')
ax.text(1.75, 2.5, 'Town10', ha='center', fontsize=11, color='white', style='italic')

arrow1 = FancyArrowPatch((3.1, 5), (4.2, 5.2),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=2.5, color=COLOR_CARLA, alpha=0.7)
arrow2 = FancyArrowPatch((3.1, 2.5), (4.2, 2.3),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=2.5, color=COLOR_CARLA, alpha=0.7)
ax.add_patch(arrow1)
ax.add_patch(arrow2)

# ==========================
# ==========================
vestibular_box = FancyBboxPatch((4.3, 4.2), 6.5, 2.3, 
                                boxstyle="round,pad=0.12", 
                                facecolor=COLOR_VESTIBULAR, 
                                edgecolor='white', 
                                linewidth=2.5, 
                                alpha=0.2)
ax.add_patch(vestibular_box)

ax.text(7.5, 6.2, 'Vestibular System', ha='center', fontsize=13, 
        weight='bold', color=COLOR_VESTIBULAR)
ax.text(7.5, 5.85, '(Half-circular canals)', ha='center', fontsize=9, 
        style='italic', color=COLOR_VESTIBULAR, alpha=0.8)

canal1 = Wedge((5.3, 5.2), 0.35, 30, 210, width=0.08, 
               facecolor=COLOR_VESTIBULAR, alpha=0.6)
canal2 = Wedge((5.8, 5.2), 0.35, 60, 240, width=0.08, 
               facecolor=COLOR_VESTIBULAR, alpha=0.6)
canal3 = Wedge((6.3, 5.2), 0.35, 90, 270, width=0.08, 
               facecolor=COLOR_VESTIBULAR, alpha=0.6)
ax.add_patch(canal1)
ax.add_patch(canal2)
ax.add_patch(canal3)

acc_box = FancyBboxPatch((7.2, 5.1), 1.2, 0.5, 
                         boxstyle="round,pad=0.05", 
                         facecolor='white', 
                         edgecolor=COLOR_VESTIBULAR, 
                         linewidth=1.5)
ang_box = FancyBboxPatch((8.7, 5.1), 1.2, 0.5, 
                         boxstyle="round,pad=0.05", 
                         facecolor='white', 
                         edgecolor=COLOR_VESTIBULAR, 
                         linewidth=1.5)
ax.add_patch(acc_box)
ax.add_patch(ang_box)
ax.text(7.8, 5.35, 'Accel.', ha='center', fontsize=9, weight='bold')
ax.text(9.3, 5.35, 'Ang. Vel.', ha='center', fontsize=9, weight='bold')

ax.text(10.3, 5.35, '(IMU)', ha='left', fontsize=8, color='gray', style='italic')

# ==========================
# ==========================
visual_box = FancyBboxPatch((4.3, 0.5), 6.5, 3.2, 
                            boxstyle="round,pad=0.12", 
                            facecolor=COLOR_VISUAL, 
                            edgecolor='white', 
                            linewidth=2.5, 
                            alpha=0.15)
ax.add_patch(visual_box)

ax.text(7.5, 3.5, 'Visual Cortex', ha='center', fontsize=13, 
        weight='bold', color=COLOR_VISUAL)
ax.text(10.2, 3.5, '(RGB)', ha='left', fontsize=8, color='gray', style='italic')

ventral_path = FancyBboxPatch((5, 2.3), 4.8, 0.7, 
                              boxstyle="round,pad=0.08", 
                              facecolor=COLOR_VENTRAL, 
                              edgecolor='white', 
                              linewidth=2, 
                              alpha=0.3)
ax.add_patch(ventral_path)
ax.text(5.3, 2.85, 'Ventral Stream', ha='left', fontsize=10, 
        weight='bold', color=COLOR_VENTRAL)
ax.text(5.3, 2.45, 'V1→V2→V4→IT', ha='left', fontsize=9, 
        color=COLOR_VENTRAL, style='italic')

for i, label in enumerate(['V1', 'V2', 'V4', 'IT']):
    x = 5.8 + i * 0.9
    circle = Circle((x, 2.65), 0.15, facecolor=COLOR_VENTRAL, 
                   edgecolor='white', linewidth=1.5, alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, 2.65, label, ha='center', va='center', 
           fontsize=7, weight='bold', color='white')

dorsal_path = FancyBboxPatch((5, 1.2), 4.8, 0.7, 
                             boxstyle="round,pad=0.08", 
                             facecolor=COLOR_DORSAL, 
                             edgecolor='white', 
                             linewidth=2, 
                             alpha=0.3)
ax.add_patch(dorsal_path)
ax.text(5.3, 1.75, 'Dorsal Stream', ha='left', fontsize=10, 
        weight='bold', color=COLOR_DORSAL)
ax.text(5.3, 1.35, 'V1→MT/MST', ha='left', fontsize=9, 
        color=COLOR_DORSAL, style='italic')

for i, label in enumerate(['V1', 'MT', 'MST']):
    x = 6.2 + i * 1.2
    circle = Circle((x, 1.55), 0.15, facecolor=COLOR_DORSAL, 
                   edgecolor='white', linewidth=1.5, alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, 1.55, label, ha='center', va='center', 
           fontsize=7, weight='bold', color='white')

# ==========================
# ==========================
fusion_diamond = Polygon([(11.3, 3.5), (12.3, 4.2), (13.3, 3.5), (12.3, 2.8)], 
                        facecolor=COLOR_FUSION, 
                        edgecolor='white', 
                        linewidth=2.5, 
                        alpha=0.7)
ax.add_patch(fusion_diamond)
ax.text(12.3, 3.5, 'Sensory\nFusion', ha='center', va='center', 
       fontsize=10, weight='bold', color='white')

arrow3 = FancyArrowPatch((10.8, 5.2), (11.8, 4.1),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color=COLOR_VESTIBULAR, alpha=0.6)
arrow4 = FancyArrowPatch((10.8, 2.3), (11.8, 2.9),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color=COLOR_VISUAL, alpha=0.6)
ax.add_patch(arrow3)
ax.add_patch(arrow4)

# ==========================
# ==========================
attention_hex = Polygon([
    (14.2, 3.5), (14.7, 4.3), (15.7, 4.3), 
    (16.2, 3.5), (15.7, 2.7), (14.7, 2.7)
], facecolor=COLOR_ATTENTION, edgecolor='white', linewidth=2.5, alpha=0.7)
ax.add_patch(attention_hex)
ax.text(15.2, 4.6, 'Spatial Attention', ha='center', fontsize=11, 
       weight='bold', color=COLOR_ATTENTION)

v1_circle = Circle((15.2, 3.5), 0.25, facecolor='yellow', 
                   edgecolor='white', linewidth=2)
ax.add_patch(v1_circle)
ax.text(15.2, 3.5, 'V1', ha='center', va='center', fontsize=9, weight='bold')

ax.text(15.2, 3.0, 'Dorsal', ha='center', fontsize=8, 
       bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_DORSAL, alpha=0.5))
ax.text(15.2, 2.5, 'Ventral', ha='center', fontsize=8, 
       bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_VENTRAL, alpha=0.5))

arrow5 = FancyArrowPatch((16.3, 3.5), (17, 3.5),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=3, color=COLOR_DECISION, alpha=0.7)
ax.add_patch(arrow5)

lstm_box = FancyBboxPatch((17.2, 4.5), 1.5, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#FFD93D', 
                          edgecolor='white', 
                          linewidth=2.5, 
                          alpha=0.9)
ax.add_patch(lstm_box)
ax.text(17.95, 5.1, 'LSTM', ha='center', va='center', 
       fontsize=12, weight='bold', color='#2C3E50')

ot_circle = Circle((17.95, 3.5), 0.5, facecolor='#6BCF7F', 
                   edgecolor='white', linewidth=2.5, alpha=0.9)
ax.add_patch(ot_circle)
ax.text(17.95, 3.5, 'OT', ha='center', va='center', 
       fontsize=11, weight='bold', color='white')

mlp_box = FancyBboxPatch((17.2, 1.5), 1.5, 1.2, 
                         boxstyle="round,pad=0.1", 
                         facecolor='#FF8C42', 
                         edgecolor='white', 
                         linewidth=2.5, 
                         alpha=0.9)
ax.add_patch(mlp_box)
ax.text(17.95, 2.1, 'MLP', ha='center', va='center', 
       fontsize=12, weight='bold', color='white')

arrow6 = FancyArrowPatch((17.95, 4.4), (17.95, 4.0),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color=COLOR_DECISION, alpha=0.6)
arrow7 = FancyArrowPatch((17.95, 3.0), (17.95, 2.7),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color=COLOR_DECISION, alpha=0.6)
ax.add_patch(arrow6)
ax.add_patch(arrow7)

output_box = FancyBboxPatch((19, 2.8), 2.5, 1.4, 
                            boxstyle="round,pad=0.12", 
                            facecolor=COLOR_DECISION, 
                            edgecolor='white', 
                            linewidth=3, 
                            alpha=0.85)
ax.add_patch(output_box)
ax.text(20.25, 3.7, 'Navigation', ha='center', fontsize=11, weight='bold', color='white')
ax.text(20.25, 3.3, 'Control', ha='center', fontsize=11, weight='bold', color='white')

arrow8 = FancyArrowPatch((18.75, 3.5), (18.95, 3.5),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=3, color=COLOR_DECISION, alpha=0.7)
ax.add_patch(arrow8)

# ==========================
# ==========================
ax.plot([0, 22], [6.8, 6.8], 'k-', linewidth=0.5, alpha=0.1)
ax.plot([0, 22], [0.2, 0.2], 'k-', linewidth=0.5, alpha=0.1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bio_nav_architecture.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'bio_nav_architecture.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 美观版生物启发导航架构图已生成！")
plt.close()
