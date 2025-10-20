import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.size': 14})

SCP_DIR = Path("results_scp_8_k4")
SDP_DIR = Path("results_sdp_8_k4")
OUTPUT_DIR = Path("results_compare_8/all")
OUTPUT_DIR.mkdir(exist_ok=True)

SCP_FILE = SCP_DIR / "final_areas.csv"
SDP_FILE = SDP_DIR / "final_areas.csv"

if not SCP_FILE.exists():
    raise FileNotFoundError(f"SCP areas file not found: {SCP_FILE}")
if not SDP_FILE.exists():
    raise FileNotFoundError(f"SDP areas file not found: {SDP_FILE}")

def load_areas(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 0:
        return np.array([float(data)], dtype=float)
    return np.asarray(data, dtype=float)

scp_areas = load_areas(SCP_FILE)
sdp_areas = load_areas(SDP_FILE)

def infer_threshold(dir_path: Path) -> float:
    thr_file = dir_path / "removal_threshold.txt"
    if thr_file.exists():
        try:
            return float(thr_file.read_text().strip())
        except Exception:
            pass
    return 1e-4

scp_thr = infer_threshold(SCP_DIR)
sdp_thr = infer_threshold(SDP_DIR)

scp_valid = scp_areas[scp_areas > scp_thr] * 1e6
sdp_valid = sdp_areas[sdp_areas > sdp_thr] * 1e6

if scp_valid.size == 0 and sdp_valid.size == 0:
    raise RuntimeError("No areas above removal threshold were found in either dataset.")

max_area = 1e4
if scp_valid.size:
    max_area = max(max_area, float(scp_valid.max()))
if sdp_valid.size:
    max_area = max(max_area, float(sdp_valid.max()))

min_threshold_mm2 = min(scp_thr, sdp_thr) * 1e6
bin_edges = np.linspace(min_threshold_mm2, max_area, 26)
bin_width = bin_edges[1] - bin_edges[0]
bar_width = bin_width * 0.45

sdp_counts, _ = np.histogram(sdp_valid, bins=bin_edges)
scp_counts, _ = np.histogram(scp_valid, bins=bin_edges)

plt.figure(figsize=(9, 6))
left_edges = bin_edges[:-1]

plt.bar(left_edges, sdp_counts, width=bar_width, align="edge",
        color="steelblue", edgecolor="black", label="SDP")
plt.bar(left_edges + bar_width, scp_counts, width=bar_width, align="edge",
        color="darkorange", edgecolor="black", label="SCP")

# Plot removal thresholds (if distinct, both will appear)
if abs(sdp_thr - scp_thr) < 1e-9:
    plt.axvline(scp_thr * 1e6, color="red", linestyle="--", linewidth=1.2, label="Removal Threshold")
else:
    plt.axvline(sdp_thr * 1e6, color="steelblue", linestyle="--", linewidth=1.2, label="SDP Threshold")
    plt.axvline(scp_thr * 1e6, color="darkorange", linestyle="--", linewidth=1.2, label="SCP Threshold")

plt.xlim(min_threshold_mm2, max_area)
plt.xticks(np.linspace(min_threshold_mm2, max_area, 11))
plt.xlabel("Cross-sectional Area (mm²)")
plt.ylabel("Number of Members")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = OUTPUT_DIR / "area_histogram_scp_vs_sdp_8k4.png"
plt.savefig(output_path, dpi=300)
print(f"Combined histogram saved to: {output_path}")
plt.show()
