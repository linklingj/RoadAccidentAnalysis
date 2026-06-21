"""Inspect scene JSON files: camera params, scale, track stats, anomalies."""
import json, sys, os, math
from collections import defaultdict, Counter

def load(f):
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)

def summarize(path):
    d = load(path)
    print("=" * 70)
    print("FILE:", os.path.basename(path))
    print("camera:", d.get("camera"))
    print("fps:", d.get("fps"), "frame_count:", d.get("frame_count"),
          "frames:", len(d.get("frames", [])), "tracks:", len(d.get("tracks", [])))
    objs = d.get("objects", [])
    print("final objects:", len(objs))
    # scale from frames
    xs, zs = [], []
    for fr in d.get("frames", []):
        for o in fr.get("objects", []):
            xs.append(o["x_m"]); zs.append(o["z_m"])
    if xs:
        print("  pos x range: %.1f .. %.1f   z range: %.1f .. %.1f" %
              (min(xs), max(xs), min(zs), max(zs)))
    # class distribution
    cls = Counter(t.get("class_name") for t in d.get("tracks", []))
    print("  track classes:", dict(cls))
    # color distribution among car tracks
    colors = Counter(t.get("color") for t in d.get("tracks", []) if t.get("class_name") == "car")
    print("  car colors:", dict(colors))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
