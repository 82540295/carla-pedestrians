# `fig/` assets

This directory holds the seven figures the paper actually compiles, the drawing sources (`.pptx`), and a few early drafts. `NeuroSLAM_KBS.tex` only references the seven PDFs in the first table; everything else here is unused.

## Referenced by the paper (7)

| Label | File | Defined in |
|---|---|---|
| `fig:3D_grid_cell_network` | `fig/3d_grid_cell_network.pdf` | `sec_method_expanded.tex` |
| `fig:ablation_and_vt` | `fig/ablation_unified.pdf` | `experiment_section.tex` |
| `fig:imu_visual_fusion` | `fig/imu_visual_complementary_fusion.pdf` | `sec_method_expanded.tex` |
| `fig:kitti_input_output` | `fig/KITTI_07_input_output_visualization.pdf` | `experiment_section.tex` |
| `fig:performance_summary` | `fig/performance_summary.pdf` | `experiment_section.tex` |
| `fig:representative_performance` | `fig/representative_performance.pdf` | `experiment_section.tex` |
| `fig:system_overview` | `fig/2_framework.pdf` | `sec_method_expanded.tex` |

## Not compiled

- `1_abstract.pdf` (0.8 MB) — graphical-abstract candidate, uploaded separately through the submission system and not typeset in the manuscript.
- `ekf_curve_KITTI07 / MH_03 / Town01 / Town02 / Town03 / Town05 / Town07 / Town10HD.pdf` and `vt_growth_all_datasets.pdf` — per-sequence curves from earlier drafts, no longer used (Town01/02 are not in the final tables either).
- Drawing sources `.pptx` (5 files, ~21.8 MB total) live in `docs/src/` and are ignored here via `neuro/kbs/fig/**/*.pptx` in `.gitignore`; rebuild the PDFs from there.

## Rebuilding

The four result figures are produced by `../regen_figures.py`, which needs the trajectories under `neuro/data/`.
The three method figures and `1_abstract.pdf` are MATLAB/PPT exports; their rebuild scripts are not shipped.
