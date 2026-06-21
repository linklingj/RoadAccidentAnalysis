"""Find track fragments that are likely the same physical object split into
multiple IDs (re-ID splits) -> merge candidates; and isolated short spurious
tracks -> delete candidates.

A merge candidate (A->B): B starts shortly after A ends (small gap or slight
overlap), endpoint positions are close, classes compatible, and the implied
speed across the gap is plausible.
"""
import sys, math
from collections import defaultdict
from bev_plot import load, pos, build_tracks

CLASS_COMPAT = [{"car","truck","bus"}, {"person","riders"}]
def compat(a, b):
    if a == b: return True
    for grp in CLASS_COMPAT:
        if a in grp and b in grp: return True
    return False

def main(scene):
    d = load(scene); tr = build_tracks(d)
    fps = d.get("fps", 30) or 30
    reg = {t["track_id"]: t for t in d.get("tracks", [])}
    info = {}
    for tid, seq in tr.items():
        frames=[f for f,_ in seq]
        info[tid]=dict(start=frames[0], end=frames[-1], n=len(seq),
                       first=pos(seq[0][1]), last=pos(seq[-1][1]),
                       cls=reg.get(tid,{}).get("class_name",""),
                       color=reg.get(tid,{}).get("color"),
                       seq=seq)
    tids=sorted(tr)
    print(f"### {scene}  ({len(tids)} tracks)")
    # merge candidates
    print("\n-- MERGE CANDIDATES (A->B: B follows A, endpoints close) --")
    cand=[]
    for a in tids:
        for b in tids:
            if a==b: continue
            A,B=info[a],info[b]
            gap = B["start"] - A["end"]   # frames between A end and B start
            if gap < -8 or gap > 25: continue       # B must start near A's end
            if B["start"] < A["start"]: continue     # B is the later one
            if not compat(A["cls"], B["cls"]): continue
            # position: A.last vs B.first
            d_ep = math.hypot(A["last"][0]-B["first"][0], A["last"][1]-B["first"][1])
            # plausible travel during gap (allow up to ~12 m/s)
            allow = max(2.0, abs(gap)/fps*12 + 1.5)
            if d_ep <= allow:
                cand.append((a,b,gap,round(d_ep,2),A["cls"],B["cls"],A["n"],B["n"]))
    for a,b,gap,d_ep,ca,cb,na,nb in sorted(cand, key=lambda x:(x[2],x[3])):
        col=f"{info[a]['color']}/{info[b]['color']}"
        print(f"   T{a}(n{na},{ca},{info[a]['color']}) -> T{b}(n{nb},{cb},{info[b]['color']})  gap={gap}fr endpoint_d={d_ep}m")
    # isolated short tracks (no merge partner)
    print("\n-- SHORT TRACKS (n<=8) --")
    partner=set()
    for a,b,*_ in cand: partner.add(a); partner.add(b)
    for tid in tids:
        if info[tid]["n"]<=8:
            tag = "" if tid in partner else "  <ISOLATED - delete?>"
            I=info[tid]
            print(f"   T{tid} n={I['n']} {I['cls']} {I['color']} frames {I['start']}..{I['end']} pos~({I['first'][0]:.1f},{I['first'][1]:.1f}){tag}")

if __name__=="__main__":
    for s in sys.argv[1:]:
        main(s); print()
