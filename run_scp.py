import sys
from pathlib import Path

# Ensure local package path
ROOT = Path(__file__).parent
PKG = ROOT / 'htto-9_8_version' / 'Sequential_Convex_Programming'
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from Sequential_Convex_Programming.cli import main

if __name__ == '__main__':
    sys.exit(main())
# sdp固定几何
# python run_scp.py --radius 5.0 --n-sectors 18 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.2 --single-subproblem --sdp-fixed-geometry --save-figs results_fixed_simple   
# sdp子问题
# python run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.5 --enable-middle-layer --middle-layer-ratio 0.7 --volume-fraction 0.3 --single-subproblem --simple-loads --save-figs results_step_simple
# scp
# python run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.5 --enable-middle-layer --middle-layer-ratio 0.7 --volume-fraction 0.5 --max-iterations 10 --simple-loads --save-figs results_scp_simple 
# python -u run_scp.py --radius 5.0 --n-sectors 10 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 200 --simple-loads --enforce-symmetry --save-figs results_scp_simple 2>&1 | tee log_scp.txt
# python -u run_scp.py --radius 5.0 --n-sectors 12 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.2 --max-iterations 30 --simple-loads --save-figs results_scp_simple 2>&1 | tee log_scp.txt
# python -u run_scp.py --radius 5.0 --n-sectors 10 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 30 --simple-loads --save-figs results_scp_simple 2>&1 | tee log_scp.txt
# 启用shell_fea
# python -u run_scp.py --radius 5.0 --n-sectors 12 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.2 --max-iterations 50 --simple-loads --enforce-symmetry --save-figs results_scp_shell 2>&1 | tee log_scp_shell.txt
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.2 --max-iterations 80 --enforce-symmetry  --save-figs results_scp_12
# --save-shell-iter
# --enable-aasi


# 10.13运行文件
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 80 --enforce-symmetry  --save-figs results_scp_8 | tee log_scp_8.txt
# python -u run_scp.py --radius 5.0 --n-sectors 12 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 80 --enforce-symmetry  --save-figs results_scp_12 | tee log_scp_12.txt
# python -u run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 80 --enforce-symmetry  --save-figs results_scp_16 | tee log_scp_16.txt
# python -u run_scp.py --radius 5.0 --n-sectors 20 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 80 --enforce-symmetry --save-figs results_scp_20 | tee log_scp_20.txt

# python run_scp.py --radius 5.0 --n-sectors 12 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_12 | tee log_sdp_12.txt
# python run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_8 | tee log_sdp_8.txt
# python run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_16 | tee log_sdp_16.txt
# python run_scp.py --radius 5.0 --n-sectors 20 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_20 | tee log_sdp_20.txt

# 10.14运行文件
# sdp
# python run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --export-shell-metrics results/shell_metrics_8_sdp.csv --shell-metrics-label sdp --save-figs results_sdp_8 | tee log_sdp_8.txt


# scp
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 80 --enforce-symmetry  --export-shell-metrics results/shell_metrics_8_scp.csv --save-figs results_scp_8 | tee log_scp_8.txt

# 10.16运行文件

# python run_scp.py --radius 5.0 --n-sectors 12 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_12 | tee log_sdp_12.txt
# python run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.2 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_8 | tee log_sdp_8.txt
# python run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_16 | tee log_sdp_16.txt
# python run_scp.py --radius 5.0 --n-sectors 20 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.1 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_20_0.1 | tee log_sdp_20_0.1.txt
# python compare_histograms.py

# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 80 --enforce-symmetry  --save-figs results_scp_8 | tee log_scp_8.txt
# python -u run_scp.py --radius 5.0 --n-sectors 12 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 80 --enforce-symmetry  --save-figs results_scp_12 | tee log_scp_12.txt
# python -u run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 80 --enforce-symmetry  --save-figs results_scp_16 | tee log_scp_16.txt
# python -u run_scp.py --radius 5.0 --n-sectors 20 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.1 --max-iterations 200 --enforce-symmetry --node-merge-threshold 0.3 --save-figs results_scp_20_0.1 | tee log_scp_20_0.1.txt


# python -u run_scp.py --radius 5.0 --rings extra_rings_4layer.json --volume-fraction 0.2 --max-iterations 80 --enforce-symmetry --save-figs results_scp_8_4 | tee log_scp_8_4.txt
# python run_scp.py --radius 5.0 --rings extra_rings_4layer.json --volume-fraction 0.2 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_8_4 | tee log_sdp_8_4.txt

# 10.18运行结果
# scp with shell
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-cap-abs 1 --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_scp_8/element_metrics.csv --save-figs results_scp_8 | tee log_scp_8.txt
# python -u run_scp.py --radius 5.0 --n-sectors 12 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-cap-abs 1 --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_scp_12/element_metrics.csv --save-figs results_scp_12 | tee log_scp_12.txt
# python -u run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-cap-abs 1 --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_scp_16/element_metrics.csv --save-figs results_scp_16 | tee log_scp_16.txt
# python -u run_scp.py --radius 5.0 --n-sectors 20 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-cap-abs 1 --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_scp_20/element_metrics.csv --save-figs results_scp_20 | tee log_scp_20.txt



# sdp with shell
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-cap-abs 1 --single-subproblem --sdp-fixed-geometry --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20   --export-element-metrics results_sdp_8/element_metrics.csv --save-figs results_sdp_8 | tee log_sdp_8.txt
# python -u run_scp.py --radius 5.0 --n-sectors 20 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-cap-abs 1 --single-subproblem --sdp-fixed-geometry --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20   --export-element-metrics results_sdp_20/element_metrics.csv --save-figs results_sdp_20 | tee log_sdp_20.txt

# 统计
# python tools/plot_element_metrics.py --scp-metrics results_scp_8/element_metrics.csv --sdp-metrics results_sdp_8/element_metrics.csv --out-dir results_compare_8 --scp-label SCP --sdp-label SDP

# 10.20运行结果
# scp problem
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --k-theta-steps 1 --middle-layer-ratio 0.8 --volume-fraction 0.2 --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_scp_8_k1/element_metrics.csv --save-figs  results_scp_8_k1 --output-dir results_scp_8_k1 | Tee-Object results_scp_8_k1/log_scp_8_k1.txt   
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --k-theta-steps 2 --middle-layer-ratio 0.8 --volume-fraction 0.2 --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_scp_8_k2/element_metrics.csv --save-figs  results_scp_8_k2 --output-dir results_scp_8_k2 | Tee-Object results_scp_8_k2/log_scp_8_k2.txt
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --k-theta-steps 3 --middle-layer-ratio 0.8 --volume-fraction 0.2 --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_scp_8_k3/element_metrics.csv --save-figs  results_scp_8_k3 --output-dir results_scp_8_k3 | Tee-Object results_scp_8_k3/log_scp_8_k3.txt
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --k-theta-steps 4 --middle-layer-ratio 0.8 --volume-fraction 0.2 --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_scp_8_k4/element_metrics.csv --save-figs  results_scp_8_k4 --output-dir results_scp_8_k4 | Tee-Object results_scp_8_k4/log_scp_8_k4.txt


# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --k-theta-steps 1 --middle-layer-ratio 0.8 --volume-fraction 0.2 --single-subproblem --sdp-fixed-geometry --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_sdp_8_k1/element_metrics.csv --save-figs  results_sdp_8_k1 --output-dir results_sdp_8_k1 | Tee-Object results_sdp_8_k1/log_sdp_8_k1.txt   
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --k-theta-steps 2 --middle-layer-ratio 0.8 --volume-fraction 0.2 --single-subproblem --sdp-fixed-geometry --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_sdp_8_k2/element_metrics.csv --save-figs  results_sdp_8_k2 --output-dir results_sdp_8_k2 | Tee-Object results_sdp_8_k2/log_sdp_8_k2.txt
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --k-theta-steps 3 --middle-layer-ratio 0.8 --volume-fraction 0.2 --single-subproblem --sdp-fixed-geometry --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_sdp_8_k3/element_metrics.csv --save-figs  results_sdp_8_k3 --output-dir results_sdp_8_k3 | Tee-Object results_sdp_8_k3/log_sdp_8_k3.txt
# 



# 数据对比
# 元素指标对比
# python tools/plot_k_multi.py --scp-metrics results_scp_8_k1/element_metrics.csv results_scp_8_k2/element_metrics.csv results_scp_8_k3/element_metrics.csv results_scp_8_k4/element_metrics.csv --sdp-metrics results_sdp_8_k1/element_metrics.csv results_sdp_8_k2/element_metrics.csv results_sdp_8_k3/element_metrics.csv results_sdp_8_k4/element_metrics.csv --k-labels 1 2 3 4 --out-dir results_compare_8/all 
# Compliance对比
# python tools/plot_compliance_multi.py --logs results_scp_8_k1/optimization_log.csv results_scp_8_k2/optimization_log.csv results_scp_8_k3/optimization_log.csv results_scp_8_k4/optimization_log.csv --labels k=1 k=2 k=3 k=4 --out-dir results_compare_8/all  
# 面积直方图对比
# python compare_histograms.py 



# sdp_fixed_problems
# python -u run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --k-theta-steps 2 --middle-layer-ratio 0.8 --volume-fraction 0.2 --max-iterations 80 --enforce-symmetry --overlay-structure-on-shell --shell-disp-unit mm --shell-disp-cbar-min-mm 0 --shell-disp-cbar-max-mm 0.50 --shell-disp-scale 20 --export-element-metrics results_scp_8_k2/element_metrics.csv --save-figs results_scp_8_k2 | tee log_scp_8_k2.txt