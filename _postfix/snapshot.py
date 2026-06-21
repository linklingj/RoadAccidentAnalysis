"""Print every active track at given frame(s): tid, class, color, pos, heading.
Usage: snapshot.py scene.json f1,f2,...
"""
import sys, math
from bev_plot import load, pos, build_tracks, sample_frame, heading

scene = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(",")]
d = load(scene); tr = build_tracks(d)
reg = {t["track_id"]: t for t in d.get("tracks", [])}
for fi in frames:
    print(f"\n=== frame {fi} ===")
    rows = []
    for tid in sorted(tr):
        seq = tr[tid]; fr = [f for f,_ in seq]
        if fi < fr[0] or fi > fr[-1]: continue
        p = sample_frame(seq, fi)
        cls = reg.get(tid,{}).get("class_name",""); col = reg.get(tid,{}).get("color")
        h = heading(seq, fi)
        hd = f"{math.degrees(h):.0f}" if h is not None else "  -"
        rows.append((cls, p[1], tid, col, p[0], hd))
    rows.sort()
    print(f"{'tid':>4} {'class':<7} {'color':<6} {'x':>7} {'z':>7} {'head':>5}")
    for cls,z,tid,col,x,hd in rows:
        print(f"{tid:>4} {cls:<7} {str(col):<6} {x:>7.2f} {z:>7.2f} {hd:>5}")
