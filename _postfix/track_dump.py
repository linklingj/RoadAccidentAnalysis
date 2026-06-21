"""Print a track's per-frame motion + heading to distinguish real turns from noise.
Usage: track_dump.py scene.json TID [stride]
"""
import sys, math
from bev_plot import load, pos, build_tracks, sample_frame, heading

scene, tid = sys.argv[1], int(sys.argv[2])
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 5
d = load(scene); tr = build_tracks(d)
seq = tr[tid]
frames = [f for f, _ in seq]
print(f"T{tid}: frames {frames[0]}..{frames[-1]}  n={len(seq)}")
print(f"{'frame':>5} {'x':>7} {'z':>7} {'stepDeg':>8} {'winHead':>8} {'dispWin':>7}")
prev = None
for i in range(0, len(seq), stride):
    f, o = seq[i]
    x, z = pos(o)
    # instantaneous heading from previous sampled point
    inst = ""
    if prev is not None:
        dx = x - prev[0]; dz = z - prev[1]
        if math.hypot(dx, dz) > 1e-4:
            inst = f"{math.degrees(math.atan2(dx, dz)):.0f}"
    wh = heading(seq, f)
    whs = f"{math.degrees(wh):.0f}" if wh is not None else "none"
    # windowed displacement
    lo = max(f-25, frames[0]); hi = f
    pa = sample_frame(seq, lo); pb = sample_frame(seq, hi)
    disp = math.hypot(pb[0]-pa[0], pb[1]-pa[1]) if pa and pb else 0
    print(f"{f:>5} {x:>7.2f} {z:>7.2f} {inst:>8} {whs:>8} {disp:>7.2f}")
    prev = (x, z)
