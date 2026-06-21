"""Apply post-correction ops to a scene JSON. Reads from _postfix/pristine/,
writes to web/data/. Idempotent (always derives from pristine).

Ops (per scene):
  delete:  [tid, ...]                      remove tracks entirely
  merge:   [[dst, [src,...]], ...]         relabel src tracks into dst
  smooth:  [[tid, window], ...]            moving-average x/z_smoothed (jitter fix)
  color:   [[tid, "black"|"white"], ...]   force a car track's colour
  trim:    [[tid, start, end], ...]        keep only frames in [start,end]
"""
import json, sys, math
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
PRISTINE = ROOT / "_postfix" / "pristine"
OUT = ROOT / "web" / "data"

def load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def pos_raw(o):
    return (o.get("x_m",0) or 0, o.get("z_m",0) or 0)

def clip_polygon(points, xmin, xmax, zmin, zmax):
    """Sutherland-Hodgman clip of a polygon ([{x,z},...]) to an axis-aligned box.
    Returns a new list of {x,z} (>=3 pts) or None if fully outside.
    Trims the road's far/sideways tail so the viewer auto-frames the action."""
    pts = [(p["x"], p["z"]) for p in points]
    def clip(pts, inside, inter):
        out = []
        n = len(pts)
        for i in range(n):
            cur = pts[i]; prev = pts[i-1]
            ci = inside(cur); pi = inside(prev)
            if ci:
                if not pi: out.append(inter(prev, cur))
                out.append(cur)
            elif pi:
                out.append(inter(prev, cur))
        return out
    def ix(a, b, xc):  # intersect segment a-b with vertical line x=xc
        t = (xc - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
        return (xc, a[1] + (b[1]-a[1])*t)
    def iz(a, b, zc):
        t = (zc - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
        return (a[0] + (b[0]-a[0])*t, zc)
    pts = clip(pts, lambda p: p[0] >= xmin, lambda a,b: ix(a,b,xmin))
    if len(pts) < 3: return None
    pts = clip(pts, lambda p: p[0] <= xmax, lambda a,b: ix(a,b,xmax))
    if len(pts) < 3: return None
    pts = clip(pts, lambda p: p[1] >= zmin, lambda a,b: iz(a,b,zmin))
    if len(pts) < 3: return None
    pts = clip(pts, lambda p: p[1] <= zmax, lambda a,b: iz(a,b,zmax))
    if len(pts) < 3: return None
    return [{"x": round(x,3), "z": round(z,3)} for x,z in pts]

def pos_sm(o):
    """Current smoothed position (Kalman output) with raw fallback."""
    sx = o.get("x_m_smoothed", 0) or 0
    sz = o.get("z_m_smoothed", 0) or 0
    if sx == 0 and sz == 0:
        return pos_raw(o)
    return (sx, sz)

def apply(fname, ops):
    d = load(PRISTINE / fname)
    delete = set(ops.get("delete", []))
    merge_map = {}
    for dst, srcs in ops.get("merge", []):
        for s in srcs: merge_map[s] = dst
    trims = {tid:(a,b) for tid,a,b in ops.get("trim", [])}
    colors = {tid:c for tid,c in ops.get("color", [])}
    smooth = {tid:w for tid,w in ops.get("smooth", [])}
    pin = set(ops.get("pin", []))

    def remap(tid):
        return merge_map.get(tid, tid)

    # 1) per-frame: delete, trim, relabel, dedupe merged, recolor
    for fr in d.get("frames", []):
        fi = fr["frame_index"]
        kept = {}
        for o in fr.get("objects", []):
            tid = o.get("track_id", -1)
            if tid in delete: continue
            if tid in trims:
                a,b = trims[tid]
                if fi < a or fi > b: continue
            ntid = remap(tid)
            o["track_id"] = ntid
            if ntid in colors: o["color"] = colors[ntid]
            # if merged collision: keep the one already present (dst-origin preferred)
            if ntid in kept:
                # prefer object whose original id == ntid (the dst), else higher conf
                if tid == ntid:
                    kept[ntid] = o
                # else keep existing
            else:
                kept[ntid] = o
        fr["objects"] = list(kept.values())

    # 1b) car render-size: the viewer derives ONE scene-wide vehicle scale from the
    # median car length_m (computeSceneVehicleScale -> rendered length ~= median).
    # length_m is depth-distorted (inflated, with horizon outliers), so cars render
    # oversized. Rescale all car length_m so the median hits car_len_target metres,
    # making rendered cars ~real size (~4.5 m) to match the footage.
    target = ops.get("car_len_target")
    if target:
        lens = []
        for fr in d["frames"]:
            for o in fr["objects"]:
                if o.get("class_name") == "car" and o.get("length_m"):
                    lens.append(o["length_m"])
        if lens:
            lens.sort()
            med = lens[len(lens)//2]
            if med > 0:
                factor = target / med
                for fr in d["frames"]:
                    for o in fr["objects"]:
                        if o.get("class_name") == "car" and o.get("length_m"):
                            o["length_m"] = round(o["length_m"] * factor, 3)
                            if o.get("width_m"):
                                o["width_m"] = round(o["width_m"] * factor, 3)
                print(f"       car length_m median {med:.2f} -> {target} (x{factor:.3f})")

    # 2) rebuild per-track sequences for smoothing / pinning
    if smooth or pin:
        seqs = defaultdict(list)
        for fr in d["frames"]:
            for o in fr["objects"]:
                seqs[o["track_id"]].append((fr["frame_index"], o))
        for tid, w in smooth.items():
            seq = sorted(seqs.get(tid, []), key=lambda x:x[0])
            if len(seq) < 3: continue
            # refine the EXISTING smoothed (Kalman) path — averaging raw is noisier
            xs = [pos_sm(o)[0] for _,o in seq]
            zs = [pos_sm(o)[1] for _,o in seq]
            half = max(1, w//2)
            nx = [sum(xs[max(0,i-half):min(len(xs),i+half+1)])/len(xs[max(0,i-half):min(len(xs),i+half+1)]) for i in range(len(xs))]
            nz = [sum(zs[max(0,i-half):min(len(zs),i+half+1)])/len(zs[max(0,i-half):min(len(zs),i+half+1)]) for i in range(len(zs))]
            for i,(fi,o) in enumerate(seq):
                o["x_m_smoothed"] = round(nx[i], 3)
                o["z_m_smoothed"] = round(nz[i], 3)
        # pin: freeze a parked car at its median position (kills spin/drift)
        for tid in pin:
            seq = sorted(seqs.get(tid, []), key=lambda x:x[0])
            if not seq: continue
            xs = sorted(pos_raw(o)[0] for _,o in seq)
            zs = sorted(pos_raw(o)[1] for _,o in seq)
            mx = xs[len(xs)//2]; mz = zs[len(zs)//2]
            for _,o in seq:
                o["x_m_smoothed"] = round(mx, 3)
                o["z_m_smoothed"] = round(mz, 3)

    # 3) registry tracks[]: drop deleted/merged-src, recolor
    new_tracks = []
    seen = set()
    for t in d.get("tracks", []):
        tid = t["track_id"]
        if tid in delete: continue
        if tid in merge_map: continue   # src folded into dst
        if tid in seen: continue
        seen.add(tid)
        if tid in colors: t["color"] = colors[tid]
        new_tracks.append(t)
    d["tracks"] = new_tracks

    # 4) rebuild objects[] snapshot from the last frame that has objects
    last_objs = []
    for fr in reversed(d["frames"]):
        if fr["objects"]:
            last_objs = [dict(o) for o in fr["objects"]]
            break
    # keep schema similar to original objects entries
    d["objects"] = last_objs

    # 4b) clip road (and far crosswalks) to the action window so the viewer's
    # auto-framing (sceneBounds-driven) focuses on the scene instead of the
    # road's far horizon tail. Vehicle data is untouched.
    cw = ops.get("clip_road")
    if cw:
        xmin, xmax, zmin, zmax = cw
        new_road = []
        for poly in d.get("road_polygons", []):
            c = clip_polygon(poly["points"], xmin, xmax, zmin, zmax)
            if c: new_road.append({"points": c})
        d["road_polygons"] = new_road
        new_cw = []
        for poly in d.get("crosswalk_polygons", []):
            c = clip_polygon(poly["points"], xmin, xmax, zmin, zmax)
            if c: new_cw.append({"points": c})
        d["crosswalk_polygons"] = new_cw
        print(f"       clip road -> x[{xmin},{xmax}] z[{zmin},{zmax}]: roads {len(new_road)}, crosswalks {len(new_cw)}")

    # 5) trajectories[]: filter to surviving ids, relabel merged
    surv = {t["track_id"] for t in d["tracks"]}
    new_tr = []
    seen_tr = set()
    for tr in d.get("trajectories", []):
        tid = remap(tr.get("track_id", -1))
        if tid in delete or tid not in surv or tid in seen_tr: continue
        seen_tr.add(tid)
        tr["track_id"] = tid
        new_tr.append(tr)
    d["trajectories"] = new_tr

    # per-scene default for the viewer's 차량 크기 slider (read by main.js applyScene)
    if ops.get("vehicle_scale") is not None:
        d["vehicle_scale"] = ops["vehicle_scale"]

    with open(OUT / fname, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print(f"[fix] {fname}: tracks {len(d['tracks'])}, frames {len(d['frames'])}, "
          f"deleted {sorted(delete)}, merged {merge_map}")

# ── Per-scene correction specs ───────────────────────────────────────────────
SCENES = {
    "sample_scene1.json": {
        # spurious short car/truck flashes + co-temporal duplicates + tiny person/riders flashes
        # 238 = white-car dup of 232; 146 = truck dup of 103; 188 = person dup on riders 37
        "delete": [80, 137, 153, 146, 188, 238, 41, 201, 261, 277, 292, 84, 115, 122],
        # re-ID fragments folded into the surviving (longer/earlier) track.
        # (136<-262 intentionally NOT merged: 262 is a stable parked car; merging
        #  would make the stopped tail wobble. They're already seamless visually.)
        "merge": [[190, [278]], [198, [326]], [251, [245]], [312, [314]]],
        "smooth": [[136, 11], [190, 11], [236, 11]],   # 228 left as-is (8.7° = mild)
        "pin": [198],   # parked car that spins ±180° from positional noise
        "car_len_target": 4.4,   # render cars at ~4.5 m (was ~5.0 m, oversized)
        "clip_road": [-10, 7, -2, 52],   # road z->99 trimmed to action (vehicles z<=47)
        "vehicle_scale": 0.17,   # default 차량 크기 slider value for this sample
    },
    "sample_scene2.json": {
        # spurious short car/person flashes; 380 = transient dup of sedan 358 during impact;
        # 342/381 = phantom riders frozen at one fixed point for the whole clip
        "delete": [368, 394, 395, 354, 359, 386, 380, 342, 381],
        # 373->344 same parked black car re-IDed; 382->379 two detections of one GAZelle truck
        "merge": [[344, [373]], [379, [382]]],
        "smooth": [[330, 11], [358, 11], [372, 11]],
        "pin": [379],   # GAZelle truck is parked post-collision; pin removes the merge seam hop
        "car_len_target": 4.4,   # render cars at ~4.5 m (was ~5.16 m, oversized)
        "clip_road": [-15, 13, -2, 56],   # road z->243 x->40 trimmed to action (vehicles z<=51 x<=10)
        "vehicle_scale": 0.23,   # default 차량 크기 slider value for this sample
    },
    "sample_scene3.json": {
        "delete": [407, 425, 439],   # spurious/duplicate short black/white flashes
        "car_len_target": 4.4,   # render cars at ~4.5 m (was ~5.49 m, oversized)
        "clip_road": [-18, 6, -2, 52],   # road z->181 x->-73 trimmed to action (vehicles z<=47 x<=-15)
        "vehicle_scale": 0.28,   # default 차량 크기 slider value for this sample
    },
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(SCENES.keys())
    for fname in targets:
        apply(fname, SCENES[fname])
