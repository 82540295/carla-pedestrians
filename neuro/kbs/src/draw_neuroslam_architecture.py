#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSLAM System Architecture Diagram
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'fig')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

color_input = '#E8F4F8'
color_visual = '#FFE5CC'
color_pose = '#D4E8D4'
color_map = '#F0E5F5'
color_output = '#FFF4CC'
color_arrow = '#4A90E2'

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
    box = FancyBboxPatch((x, y), w, h, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', 
                         facecolor=color, 
                         linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, 
            ha='center', va='center', 
            fontsize=fontsize, weight=weight,
            wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', color=color_arrow):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           mutation_scale=30, 
                           linewidth=2.5,
                           color=color)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, 
                ha='center', va='bottom',
                fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor='none', 
                         alpha=0.8))

ax.text(8, 9.5, 'NeuroSLAM System Architecture', 
        ha='center', va='top', fontsize=20, weight='bold')
ax.text(8, 9.0, 'Brain-Inspired 4DoF SLAM with Enhanced Visual Template', 
        ha='center', va='top', fontsize=12, style='italic', color='gray')

# RGB Camera
draw_box(ax, 0.5, 0.5, 2.5, 1.2, 
         'RGB Camera\n120×160\n10 FPS', 
         color_input, fontsize=10)

# IMU Sensor
draw_box(ax, 3.5, 0.5, 2.5, 1.2, 
         'IMU Sensor\nAccel + Gyro\nSelf-motion', 
         color_input, fontsize=10)

draw_box(ax, 0.5, 2.5, 2.5, 1.0, 
         'Preprocessing\nGrayscale\nCLAHE\nGaussian', 
         color_visual, fontsize=9)

draw_box(ax, 0.5, 4.0, 2.5, 1.2, 
         'Feature Extraction\nV1-IT Pathway\nRow Normalization\n64×64', 
         color_visual, fontsize=9)

draw_box(ax, 0.5, 5.7, 2.5, 1.0, 
         'VT Matching\nCosine Distance\nThreshold=0.07', 
         color_visual, fontsize=9)

ax.text(1.75, 7.0, 'Enhanced Visual Template Module', 
        ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', 
                 facecolor=color_visual, 
                 edgecolor='black', 
                 linewidth=1.5))

# 3D Grid Cells
draw_box(ax, 6.5, 3.0, 3.0, 2.0, 
         '3D Grid Cell Network\n(x, y, z)\n\n• Attractor Dynamics\n• Path Integration\n• Local View Calibration', 
         color_pose, fontsize=9)

# Head Direction Cells
draw_box(ax, 10.0, 3.0, 3.0, 2.0, 
         'Multilayered HDC Network\n(yaw)\n\n• 2D MD-CAN\n• Layer-wise Direction\n• Rotation Update', 
         color_pose, fontsize=9)

ax.text(8.75, 5.5, 'Conjunctive Pose Cell Network', 
        ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', 
                 facecolor=color_pose, 
                 edgecolor='black', 
                 linewidth=1.5))

# Experience Map
draw_box(ax, 13.5, 2.0, 2.0, 2.5, 
         'Multilayered\nExperience Map\n\nNodes:\n(VT, GC, HDC)\n\nLinks:\nΔpose', 
         color_map, fontsize=9)

# Loop Closure
draw_box(ax, 13.5, 5.0, 2.0, 1.0, 
         'Loop Closure\n& Relaxation', 
         color_map, fontsize=9)

ax.text(14.5, 6.5, 'Experience Map', 
        ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', 
                 facecolor=color_map, 
                 edgecolor='black', 
                 linewidth=1.5))

draw_box(ax, 6.5, 7.5, 3.0, 1.0, 
         '4DoF Pose Estimate\n(x, y, z, yaw)', 
         color_output, fontsize=10, bold=True)

draw_box(ax, 10.0, 7.5, 3.0, 1.0, 
         '3D Topological Map\nwith Loop Closure', 
         color_output, fontsize=10, bold=True)

# RGB -> Preprocessing
draw_arrow(ax, 1.75, 1.7, 1.75, 2.5, 'Image')

# Preprocessing -> Feature Extraction
draw_arrow(ax, 1.75, 3.5, 1.75, 4.0, 'Enhanced\nFeatures')

# Feature Extraction -> VT Matching
draw_arrow(ax, 1.75, 5.2, 1.75, 5.7, 'Feature\nVector')

# VT -> 3D Grid Cells
draw_arrow(ax, 3.0, 6.2, 6.5, 4.5, 'VT ID')

# VT -> Experience Map
draw_arrow(ax, 3.0, 6.2, 13.5, 3.5, 'Local View')

# IMU -> 3D Grid Cells
draw_arrow(ax, 4.75, 1.7, 7.5, 3.0, 'Velocity')

# IMU -> HDC
draw_arrow(ax, 5.0, 1.7, 11.0, 3.0, 'Rotation')

# 3D Grid Cells -> Output Pose
draw_arrow(ax, 8.0, 5.0, 8.0, 7.5, 'Position\n(x,y,z)')

# HDC -> Output Pose
draw_arrow(ax, 11.5, 5.0, 11.5, 7.5, 'Orientation\n(yaw)')

# 3D Grid Cells -> Experience Map
draw_arrow(ax, 9.5, 4.0, 13.5, 3.0, 'GC Activity')

# HDC -> Experience Map
draw_arrow(ax, 13.0, 4.0, 13.5, 3.5, 'HDC Activity')

# Experience Map -> Loop Closure
draw_arrow(ax, 14.5, 4.5, 14.5, 5.0, 'Match')

# Loop Closure -> Experience Map (feedback)
draw_arrow(ax, 14.0, 5.0, 14.0, 4.5, 'Correct', 'red')

# Experience Map -> Output Map
draw_arrow(ax, 14.5, 6.0, 11.5, 7.5, 'Map')

# Loop Closure -> Pose Cells (feedback)
ax.annotate('', xy=(9.5, 5.0), xytext=(13.5, 5.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='red', 
                          connectionstyle="arc3,rad=.3"))
ax.text(11.5, 6.0, 'Calibration', ha='center', fontsize=8, 
        style='italic', color='red')

legend_elements = [
    mpatches.Patch(color=color_input, label='Sensor Input', edgecolor='black'),
    mpatches.Patch(color=color_visual, label='Visual Processing', edgecolor='black'),
    mpatches.Patch(color=color_pose, label='Pose Representation', edgecolor='black'),
    mpatches.Patch(color=color_map, label='Spatial Memory', edgecolor='black'),
    mpatches.Patch(color=color_output, label='System Output', edgecolor='black'),
    mlines.Line2D([], [], color=color_arrow, marker='>', markersize=8, 
                  label='Forward Flow', linewidth=2),
    mlines.Line2D([], [], color='red', marker='>', markersize=8, 
                  label='Feedback', linewidth=2),
]
ax.legend(handles=legend_elements, loc='lower center', 
          bbox_to_anchor=(0.5, -0.05), ncol=7, 
          frameon=True, fontsize=10)

ax.text(0.2, 6.8, '★', fontsize=20, color='red', weight='bold')
ax.text(0.5, 6.5, 'Innovation 1:\n5880% more\nVT recognition', 
        fontsize=7, color='darkred',
        bbox=dict(boxstyle='round,pad=0.3', 
                 facecolor='yellow', alpha=0.7))

ax.text(6.2, 5.2, '★', fontsize=20, color='red', weight='bold')
ax.text(3.8, 5.0, 'Innovation 2:\n4DoF Pose\nRepresentation', 
        fontsize=7, color='darkred',
        bbox=dict(boxstyle='round,pad=0.3', 
                 facecolor='yellow', alpha=0.7))

ax.text(15.8, 5.5, '★', fontsize=20, color='red', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'neuroslam_architecture.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'neuroslam_architecture.png'),
            dpi=300, bbox_inches='tight')
print("✅ Architecture diagram saved:")
print("   - neuroslam_architecture.pdf")
print("   - neuroslam_architecture.png")
plt.show()
