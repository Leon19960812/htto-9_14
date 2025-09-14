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
# python run_scp.py --radius 2.0 --n-sectors 12 --inner-ratio 0.6 --depth 10 --volume-fraction 0.2 --max-iterations 10 --save-figs results