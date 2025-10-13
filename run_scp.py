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
# python -u run_scp.py --radius 5.0 --n-sectors 18 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --max-iterations 80 --enforce-symmetry --save-figs results_scp_18 | tee log_scp_18.txt


# python run_scp.py --radius 5.0 --n-sectors 12 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_12
# python run_scp.py --radius 5.0 --n-sectors 8 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_8
# python run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_16
# python run_scp.py --radius 5.0 --n-sectors 18 --inner-ratio 0.6 --enable-middle-layer --middle-layer-ratio 0.8 --volume-fraction 0.3 --single-subproblem --sdp-fixed-geometry --save-figs results_sdp_18