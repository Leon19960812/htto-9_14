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
# python run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.5 --enable-middle-layer --middle-layer-ratio 0.7 --volume-fraction 0.5 --single-subproblem --sdp-fixed-geometry --simple-loads --sdp-reg-eps 1e-8 --sdp-lmi-eps 1e-7 --sdp-verbose --save-figs results_fixed_simple   
# sdp子问题
# python run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.5 --enable-middle-layer --middle-layer-ratio 0.7 --volume-fraction 0.3 --single-subproblem --simple-loads --save-figs results_step_simple
# scp
# python run_scp.py --radius 5.0 --n-sectors 16 --inner-ratio 0.5 --enable-middle-layer --middle-layer-ratio 0.7 --volume-fraction 0.3 --max-iterations 10 --simple-loads --save-figs results_scp_simple 