"""Render a precise top-down (BEV) map of a scene JSON, replicating the viewer's
heading algorithm, so anomalies (duplicate boxes, flashing, wrong heading,
positions) are directly visible.

Usage:
  bev_plot.py scene.json overview out.png          # all trajectories
  bev_plot.py scene.json frames 100,200,300 out.png # oriented boxes @ frames
"""
import json, sys, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle
import matplotlib.transforms as mtransforms

def load(f):
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)

def pos(o):
    sx = o.get("x_m_smoothed", 0) or 0
    sz = o.get("z_m_smoothed", 0) or 0
    if sx == 0 and sz == 0:
        return (o.get("x_m", 0) or 0, o.get("z_m", 0) or 0)
    return (sx, sz)

def build_tracks(d):
    tr = defaultdict(list)
    for fr in d.get("frames", []):
        fi = fr["frame_index"]
        for o in fr.get("objects", []):
            tid = o.get("track_id", -1)
            if tid < 0: continue
            tr[tid].append((fi, o))
    for tid in tr: tr[tid].sort(key=lambda x: x[0])
    return tr

def sample_frame(seq, fi):
    """Position at integer/float frame via lerp between bracketing recorded frames."""
    frames = [f for f,_ in seq]
    ps = [pos(o) for _,o in seq]
    if not frames: return None
    if fi <= frames[0]: return ps[0]
    if fi >= frames[-1]: return ps[-1]
    lo, hi = 0, len(frames)-1
    while lo+1 < hi:
        mid=(lo+hi)//2
        if frames[mid] <= fi: lo=mid
        else: hi=mid
    if frames[lo]==fi: return ps[lo]
    a=(fi-frames[lo])/(frames[hi]-frames[lo])
    return (ps[lo][0]+(ps[hi][0]-ps[lo][0])*a, ps[lo][1]+(ps[hi][1]-ps[lo][1])*a)

def heading(seq, frameF, window=25, samples=16, mindisp=0.05, outlier_deg=45):
    """Replicate playback.js _computeHeading; returns angle (atan2(dx,dz)) or None."""
    frames=[f for f,_ in seq]
    start,end=frames[0],frames[-1]
    lo=max(frameF-window, start); hi=min(frameF, end)
    if hi-lo < 1e-3: return None
    pts=[]
    for s in range(samples):
        f=lo+(hi-lo)*(s/(samples-1))
        p=sample_frame(seq,f)
        if p: pts.append(p)
    if len(pts)<2: return None
    disp=math.hypot(pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1])
    if disp<mindisp: return None
    ang=[]; mag=[]
    for i in range(1,len(pts)):
        dx=pts[i][0]-pts[i-1][0]; dz=pts[i][1]-pts[i-1][1]
        m2=dx*dx+dz*dz
        if m2<1e-8: continue
        ang.append(math.atan2(dx,dz)); mag.append(math.sqrt(m2))
    if not ang: return None
    ref=ang[0]
    wrapped=[]
    for a in ang:
        dd=a-ref
        while dd>math.pi: dd-=2*math.pi
        while dd<-math.pi: dd+=2*math.pi
        wrapped.append(ref+dd)
    med=sorted(wrapped)[len(wrapped)//2]
    thr=math.radians(outlier_deg)
    ss=cc=0.0
    for i in range(len(ang)):
        if abs(wrapped[i]-med)>thr: continue
        ss+=math.sin(ang[i])*mag[i]; cc+=math.cos(ang[i])*mag[i]
    if ss==0 and cc==0: ss=math.sin(med); cc=math.cos(med)
    return math.atan2(ss,cc)

CARSZ={"car":(4.5,2.0),"truck":(7.5,2.6),"bus":(10,2.8),"person":(0.6,0.6),"riders":(1.8,0.8)}

def draw_box(ax, x, z, ang, cls, color, tid):
    L,W=CARSZ.get(cls,(2,2))
    if cls in ("person","riders"):
        fc = "magenta" if cls=="person" else "orange"
        ax.plot([x],[z],"o",color=fc,ms=4,zorder=5)
        ax.text(x,z,str(tid),fontsize=5,color=fc,zorder=6)
        return
    fc = {"black":"#222","white":"#eee"}.get(color,"#3a7")
    deg = math.degrees(ang if ang is not None else 0)
    r=Rectangle((-L/2,-W/2),L,W,facecolor=fc,edgecolor="red" if ang is None else "lime",lw=0.8,alpha=0.85,zorder=4)
    t=mtransforms.Affine2D().rotate_deg(-deg).translate(x,z)+ax.transData
    r.set_transform(t); ax.add_patch(r)
    # heading tick (front)
    if ang is not None:
        fx=x+math.sin(ang)*L*0.6; fz=z+math.cos(ang)*L*0.6
        ax.plot([x,fx],[z,fz],"-",color="cyan",lw=1,zorder=5)
    ax.text(x,z,str(tid),fontsize=6,color="red",ha="center",va="center",zorder=7,weight="bold")

def plot_overview(d, out):
    tr=build_tracks(d)
    reg={t["track_id"]:t for t in d.get("tracks",[])}
    fig,ax=plt.subplots(figsize=(10,12))
    for poly in d.get("road_polygons",[]):
        pts=[(p["x"],p["z"]) for p in poly["points"]]
        ax.add_patch(MplPoly(pts,closed=True,facecolor="#ddd",edgecolor="#999",alpha=0.4,zorder=0))
    cmap=plt.cm.tab20
    for i,tid in enumerate(sorted(tr)):
        seq=tr[tid]; cls=reg.get(tid,{}).get("class_name","")
        xs=[pos(o)[0] for _,o in seq]; zs=[pos(o)[1] for _,o in seq]
        c=cmap(i%20)
        ax.plot(xs,zs,"-",color=c,lw=1,alpha=0.7)
        ax.plot(xs[0],zs[0],"o",color=c,ms=3)
        ax.text(xs[0],zs[0],f"T{tid}",fontsize=6,color=c)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"Overview {len(tr)} tracks"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out,dpi=90); print("wrote",out)

def plot_frames(d, frame_list, out):
    tr=build_tracks(d)
    reg={t["track_id"]:t for t in d.get("tracks",[])}
    n=len(frame_list)
    cols=min(n,3); rows=(n+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(6*cols,7*rows),squeeze=False)
    # global bounds
    allx=[pos(o)[0] for seq in tr.values() for _,o in seq]
    allz=[pos(o)[1] for seq in tr.values() for _,o in seq]
    xmin,xmax=min(allx)-3,max(allx)+3; zmin,zmax=min(allz)-3,max(allz)+3
    for k,fi in enumerate(frame_list):
        ax=axes[k//cols][k%cols]
        for poly in d.get("road_polygons",[]):
            pts=[(p["x"],p["z"]) for p in poly["points"]]
            ax.add_patch(MplPoly(pts,closed=True,facecolor="#e8e8e8",edgecolor="#aaa",alpha=0.5,zorder=0))
        for tid in sorted(tr):
            seq=tr[tid]; frames=[f for f,_ in seq]
            if fi<frames[0] or fi>frames[-1]: continue
            p=sample_frame(seq,fi)
            if not p: continue
            cls=reg.get(tid,{}).get("class_name","")
            color=reg.get(tid,{}).get("color")
            ang=heading(seq,fi)
            draw_box(ax,p[0],p[1],ang,cls,color,tid)
        ax.set_xlim(xmin,xmax); ax.set_ylim(zmax,zmin)
        ax.set_aspect("equal"); ax.set_title(f"frame {fi}"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out,dpi=85); print("wrote",out)

if __name__=="__main__":
    scene=sys.argv[1]; mode=sys.argv[2]; d=load(scene)
    if mode=="overview":
        plot_overview(d, sys.argv[3])
    elif mode=="frames":
        fl=[int(x) for x in sys.argv[3].split(",")]
        plot_frames(d, fl, sys.argv[4])
