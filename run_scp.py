import argparse
import json
import sys
from pathlib import Path

# Ensure local package path
ROOT = Path(__file__).parent
PKG = ROOT / 'htto-9_8_version' / 'Sequential_Convex_Programming'
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from scp_optimizer import SequentialConvexTrussOptimizer


def parse_args():
    p = argparse.ArgumentParser(description='Run SCP truss optimization with PolarGeometry ground structure')
    p.add_argument('--radius', type=float, default=2.0)
    p.add_argument('--n-sectors', type=int, default=12)
    p.add_argument('--inner-ratio', type=float, default=0.7)
    p.add_argument('--depth', type=float, default=50.0)
    p.add_argument('--volume-fraction', type=float, default=0.1)
    p.add_argument('--rings', type=str, default=None, help='Path to rings JSON file')
    p.add_argument('--enable-aasi', action='store_true', help='Enable AASI constraints phase C')
    return p.parse_args()


def load_rings(path: str, radius: float, n_sectors: int, inner_ratio: float):
    if not path:
        # default two-ring template
        return [
            {"radius": radius, "n_nodes": n_sectors + 1, "type": "outer"},
            {"radius": radius * inner_ratio, "n_nodes": n_sectors + 1, "type": "inner"},
        ]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError('rings file must contain a JSON array')
        return data


def main():
    args = parse_args()
    rings = load_rings(args.rings, args.radius, args.n_sectors, args.inner_ratio)

    opt = SequentialConvexTrussOptimizer(
        radius=args.radius,
        n_sectors=args.n_sectors,
        inner_ratio=args.inner_ratio,
        depth=args.depth,
        volume_fraction=args.volume_fraction,
        enable_aasi=args.enable_aasi,
        polar_rings=rings,
    )
    opt.solve_scp_optimization()

    # Print summary
    print('\nOptimization finished.')
    if getattr(opt, 'final_compliance', None) is not None:
        print(f'Final compliance: {opt.final_compliance:.6e}')


if __name__ == '__main__':
    main()

