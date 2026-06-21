"""Classify car/truck tracks as PARKED (pin), MOVING+jittery (smooth), or OK.
Prints suggested pin/smooth lists.
"""
import sys, math
from collections import Counter
from bev_plot import load, pos, build_tracks, heading

def jitter_of(seq):
    angs=[]
    for i in range(1,len(seq)):
        x0,z0=pos(seq[i-1][1]); x1,z1=pos(seq[i][1])
        dx,dz=x1-x0,z1-z0
        if math.hypot(dx,dz)>0.05: angs.append(math.atan2(dx,dz))
    if len(angs)<3: return 0.0
    s=sum(math.sin(a) for a in angs)/len(angs); c=sum(math.cos(a) for a in angs)/len(angs)
    R=math.hypot(s,c)
    return math.degrees(math.sqrt(max(0,-2*math.log(max(R,1e-9)))))

def main(scene):
    d=load(scene); tr=build_tracks(d)
    reg={t["track_id"]:t for t in d.get("tracks",[])}
    pin=[]; smooth=[]
    print(f"### {scene}")
    print(f"{'tid':>4} {'cls':<6} {'n':>4} {'netD':>5} {'maxR':>5} {'pathL':>6} {'jit':>5}  verdict")
    for tid in sorted(tr):
        seq=tr[tid]; cls=reg.get(tid,{}).get("class_name","")
        if cls not in ("car","truck","bus"): continue
        xs=[pos(o)[0] for _,o in seq]; zs=[pos(o)[1] for _,o in seq]
        n=len(seq)
        netD=math.hypot(xs[-1]-xs[0], zs[-1]-zs[0])
        cx=sum(xs)/n; cz=sum(zs)/n
        maxR=max(math.hypot(x-cx,z-cz) for x,z in zip(xs,zs))
        pathL=sum(math.hypot(xs[i]-xs[i-1],zs[i]-zs[i-1]) for i in range(1,n))
        jit=jitter_of(seq)
        verdict="ok"
        if n>=30 and maxR<1.5 and netD<1.5:
            verdict="PARKED->pin"; pin.append(tid)
        elif jit>8 and netD>=1.5:
            verdict="MOVING jittery->smooth"; smooth.append((tid,11))
        elif jit>10 and n>=20:
            # jittery but not clearly moving nor tightly parked: smooth medium
            verdict="jittery->smooth"; smooth.append((tid,13))
        print(f"{tid:>4} {cls:<6} {n:>4} {netD:>5.1f} {maxR:>5.1f} {pathL:>6.1f} {jit:>5.1f}  {verdict}")
    print("\nPIN =", pin)
    print("SMOOTH =", smooth)

if __name__=="__main__":
    for s in sys.argv[1:]: main(s); print()
