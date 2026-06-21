"""Detect anomalies in a scene JSON's per-frame track timeline.

Flags, per track: flashing (too few frames), sparse coverage (many gaps),
teleports (implausible speed), large interior gaps, color inconsistency,
mid-scene pop-in/out. Cross-track: duplicate detections (co-temporal & close).

Heading is derived by the viewer from the motion trend of x_m_smoothed/z_m_smoothed,
so position jitter == bad heading; we also report heading jitter.
"""
import json, sys, math
from collections import defaultdict, Counter

def load(f):
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)

def build_tracks(d):
    """track_id -> sorted list of (frame_index, obj)."""
    tr = defaultdict(list)
    for fr in d.get("frames", []):
        fi = fr["frame_index"]
        for o in fr.get("objects", []):
            tid = o.get("track_id", -1)
            if tid < 0:
                continue
            tr[tid].append((fi, o))
    for tid in tr:
        tr[tid].sort(key=lambda x: x[0])
    return tr

def pos(o):
    sx = o.get("x_m_smoothed", 0) or 0
    sz = o.get("z_m_smoothed", 0) or 0
    if sx == 0 and sz == 0:
        return (o.get("x_m", 0) or 0, o.get("z_m", 0) or 0)
    return (sx, sz)

def analyze(path):
    d = load(path)
    fps = d.get("fps", 30) or 30
    tr = build_tracks(d)
    reg = {t["track_id"]: t for t in d.get("tracks", [])}
    nframes_total = len(d.get("frames", []))
    print("#" * 78)
    print(f"# {path}")
    print(f"# fps={fps:.2f} frames={nframes_total} tracks={len(tr)} registry={len(reg)}")
    print("#" * 78)

    flash, sparse, teleport, gaps, colorbad, popinout = [], [], [], [], [], []
    rows = []
    for tid in sorted(tr):
        seq = tr[tid]
        frames = [fi for fi, _ in seq]
        start, end = frames[0], frames[-1]
        span = end - start + 1
        n = len(seq)
        coverage = n / span if span else 1.0
        cls = reg.get(tid, {}).get("class_name") or seq[0][1].get("class_name", "")
        regcolor = reg.get(tid, {}).get("color")
        # color distribution per frame
        cols = Counter(o.get("color") for _, o in seq if o.get("color"))
        # speeds (per consecutive recorded sample, normalized to m/s)
        maxstep = 0.0; maxstep_at = None; speeds = []
        interior_gaps = []
        for i in range(1, n):
            (f0, o0), (f1, o1) = seq[i-1], seq[i]
            df = f1 - f0
            if df > 1:
                interior_gaps.append((f0, f1, df))
            x0, z0 = pos(o0); x1, z1 = pos(o1)
            dist = math.hypot(x1-x0, z1-z0)
            mps = dist / (df / fps) if df else 0
            speeds.append(mps)
            step = dist / df if df else dist  # per-frame
            if step > maxstep:
                maxstep = step; maxstep_at = f0
        meanspeed = sum(speeds)/len(speeds) if speeds else 0
        # heading jitter: stddev of per-step heading angle (only steps with motion)
        angs = []
        for i in range(1, n):
            (f0, o0), (f1, o1) = seq[i-1], seq[i]
            x0, z0 = pos(o0); x1, z1 = pos(o1)
            dx, dz = x1-x0, z1-z0
            if math.hypot(dx, dz) > 0.05:
                angs.append(math.atan2(dx, dz))
        # circular stdev (deg)
        if len(angs) >= 3:
            s = sum(math.sin(a) for a in angs)/len(angs)
            c = sum(math.cos(a) for a in angs)/len(angs)
            R = math.hypot(s, c)
            jitter = math.degrees(math.sqrt(max(0, -2*math.log(max(R,1e-9)))))
        else:
            jitter = 0.0

        rows.append((tid, cls, regcolor, dict(cols), start, end, span, n,
                     round(coverage,2), round(meanspeed,1), round(maxstep,2),
                     maxstep_at, round(jitter,1), interior_gaps))

        # flags
        if n <= 6:
            flash.append(tid)
        elif coverage < 0.5 and span > 12:
            sparse.append(tid)
        if maxstep > 2.5:  # >2.5 m in a single frame ~ >75 m/s
            teleport.append((tid, maxstep, maxstep_at))
        big = [g for g in interior_gaps if g[2] >= 8]
        if big:
            gaps.append((tid, big))
        if cls == "car" and len(cols) >= 2:
            # color flip-flop (registry should be majority; flag if minority is sizable)
            tot = sum(cols.values()); top = cols.most_common(1)[0][1]
            if tot - top >= 3:
                colorbad.append((tid, dict(cols), regcolor))
        # mid-scene pop-in/out: starts after frame 5 and ends before last-5, with decent length
        if start > 8 and end < nframes_total - 8 and n >= 8:
            popinout.append((tid, start, end, n))

    # print table
    print(f"{'tid':>4} {'class':<7} {'col':<6} {'start':>5} {'end':>5} {'n':>4} {'cov':>4} {'spd':>5} {'mxstep':>6} {'jit':>5}  colors")
    for (tid, cls, regcolor, cols, start, end, span, n, cov, spd, mx, mxat, jit, ig) in rows:
        cstr = ",".join(f"{k}:{v}" for k,v in cols.items()) if cols else "-"
        gstr = f" GAPS={[(a,b,df) for a,b,df in ig if df>=8]}" if any(df>=8 for _,_,df in ig) else ""
        print(f"{tid:>4} {cls:<7} {str(regcolor):<6} {start:>5} {end:>5} {n:>4} {cov:>4} {spd:>5} {mx:>6} {jit:>5}  {cstr}{gstr}")

    # cross-track duplicates
    dups = []
    tids = sorted(tr)
    for i in range(len(tids)):
        for j in range(i+1, len(tids)):
            a, b = tids[i], tids[j]
            ca = reg.get(a,{}).get("class_name"); cb = reg.get(b,{}).get("class_name")
            # build frame->pos maps
            pa = {fi: pos(o) for fi,o in tr[a]}
            pb = {fi: pos(o) for fi,o in tr[b]}
            common = sorted(set(pa) & set(pb))
            if len(common) < 8:
                continue
            dists = [math.hypot(pa[f][0]-pb[f][0], pa[f][1]-pb[f][1]) for f in common]
            dists.sort()
            med = dists[len(dists)//2]
            if med < 2.0:
                dups.append((a, b, ca, cb, len(common), round(med,2), round(min(dists),2), round(max(dists),2)))

    print("\n=== FLAGS ===")
    print("flash (n<=6):", flash)
    print("sparse (cov<0.5):", sparse)
    print("teleport (>2.5m/frame):", [(t, round(m,2), at) for t,m,at in teleport])
    print("interior gaps>=8fr:", [(t, g) for t,g in gaps])
    print("color-flip cars:", colorbad)
    print("mid-scene pop-in/out:", [(t,s,e,n) for t,s,e,n in popinout])
    print("DUPLICATE PAIRS (median dist<2m, >=8 common frames):")
    for a,b,ca,cb,nc,med,mn,mx in sorted(dups, key=lambda x:x[5]):
        print(f"   T{a}({ca}) ~ T{b}({cb}): common={nc} median={med}m min={mn} max={mx}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
