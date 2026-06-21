"""Measure what inflates the viewer's auto-framing bounds per scene:
road polygon extent, crosswalk extent, vehicle activity percentiles, and
far-outlier tracks (median position far beyond the bulk of activity).
"""
import json, sys, statistics
from collections import defaultdict

def pos(o):
    sx=o.get("x_m_smoothed",0) or 0; sz=o.get("z_m_smoothed",0) or 0
    if sx==0 and sz==0: return (o.get("x_m",0) or 0, o.get("z_m",0) or 0)
    return (sx,sz)

def pct(vals,p):
    vals=sorted(vals); k=(len(vals)-1)*p/100; f=int(k)
    return vals[f] if f+1>=len(vals) else vals[f]+(vals[f+1]-vals[f])*(k-f)

def extent(name, polys):
    xs=[p["x"] for poly in polys for p in poly["points"]]
    zs=[p["z"] for poly in polys for p in poly["points"]]
    if not xs: print(f"  {name}: none"); return
    print(f"  {name}: n={len(polys)}  x[{min(xs):.1f},{max(xs):.1f}]  z[{min(zs):.1f},{max(zs):.1f}]")

for f in sys.argv[1:]:
    d=json.load(open(f,encoding="utf-8"))
    print("="*70); print(f)
    extent("road", d.get("road_polygons",[]))
    extent("crosswalks", d.get("crosswalk_polygons",[]))
    # vehicle activity
    allx=[]; allz=[]; tr=defaultdict(list)
    for fr in d["frames"]:
        for o in fr["objects"]:
            x,z=pos(o); allx.append(x); allz.append(z)
            tr[o["track_id"]].append((x,z))
    print(f"  vehicles: x[{min(allx):.1f},{max(allx):.1f}]  z[{min(allz):.1f},{max(allz):.1f}]")
    print(f"     x pct 2/50/98: {pct(allx,2):.1f}/{pct(allx,50):.1f}/{pct(allx,98):.1f}")
    print(f"     z pct 2/50/98: {pct(allz,2):.1f}/{pct(allz,50):.1f}/{pct(allz,98):.1f}")
    z98=pct(allz,98); x2=pct(allx,2); x98=pct(allx,98)
    # far-outlier tracks (median z beyond z98 by margin, or median x outside x range)
    reg={t["track_id"]:t for t in d.get("tracks",[])}
    outliers=[]
    for tid,pts in tr.items():
        mz=statistics.median(z for _,z in pts); mx=statistics.median(x for x,_ in pts)
        if mz > z98+3 or mx < x2-3 or mx > x98+3:
            outliers.append((tid, reg.get(tid,{}).get("class_name"), round(mx,1), round(mz,1), len(pts)))
    print(f"     far-outlier tracks (beyond bulk): {sorted(outliers, key=lambda t:-t[3])}")
