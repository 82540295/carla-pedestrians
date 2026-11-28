#!/bin/bash
# ============================================================================
# grid_search_gates.sh — coordinate grid search over the HART gate constants (2026-11-16)
#
#
# ============================================================================
SCENE="${1:-TOWN03}"
PARALLEL="${2:-1}"
AXES=(INPUT FORGET OUTPUT)
VALS=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
DEFAULTS=(0.5 0.5 0.9)
CSV="$(cd "$(dirname "$0")" && pwd)/grid_search_results.csv"
echo "axis,value,rmse,log" > "$CSV"

get_axis_idx() { case "$1" in INPUT) echo 0;; FORGET) echo 1;; OUTPUT) echo 2;; esac; }

set_gates() {  # $1=axis  $2=value
  case "$1" in
    INPUT)  export HART_INPUT_GATE="$2";;
    FORGET) export HART_FORGET_GATE="$2";;
    OUTPUT) export HART_OUTPUT_GATE="$2";;
  esac
}

run_one() {  # $1=axis $2=value
  local axis="$1" val="$2"
  if grep -q "^$axis,$val,[0-9]" "$CSV" 2>/dev/null; then
    echo "[GS] 跳过已完成 $axis=$val"
    return
  fi
  local tag="${axis}_${val}"
  local log="/tmp/gs_${SCENE}_${tag//./_}.log"
  echo "[GS] 跑 $axis=$val ..."
  ~/matlab_batch.sh "RERUN_${SCENE}" > "$log" 2>&1
  local rc=$?
  local rmse
  local report="$(cd "$(dirname "$0")/../.." && pwd)/data/01_NeuroSLAM_Datasets/${SCENE}Data_IMU_Fusion/slam_results/performance_report.txt"
  cp "$report" "/tmp/gs_report_${SCENE}_${tag//./_}.txt" 2>/dev/null  # 组合报告快照(事后取证)
  rmse=$(grep -A6 "Experience Map:" "$report" 2>/dev/null | grep -m1 "RMSE" | grep -oE "[0-9]+\.[0-9]+")
  if [ -z "$rmse" ]; then
    rmse=$(grep "RMSE:" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    echo "[GS]   (报告无 Experience 块, 用日志末段 RMSE=$rmse)"
  fi
  echo "$axis,$val,$rmse,$log" >> "$CSV"
  echo "[GS] $axis=$val → RMSE=${rmse:-解析失败} (exit=$rc)"
}

pids=()
for axis in "${AXES[@]}"; do
  for v in "${VALS[@]}"; do
    set_gates "$axis" "$v"
    if [ "$PARALLEL" = "2" ]; then
      run_one "$axis" "$v" & pids+=($!)
      while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
    else
      run_one "$axis" "$v"
    fi
  done
done
wait

echo "==== 各轴最优 ===="
for axis in "${AXES[@]}"; do
  awk -F, -v a="$axis" '$1==a && $3!="" && $3!="解析失败" {print $3, $2}' "$CSV" \
    | sort -n | head -1 | awk -v a="$axis" '{print a" 最优: "$2" (RMSE="$1")"}'
done
echo "完整表见 $CSV"
