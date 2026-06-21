"""Side-by-side ground-truth diagnostic: original video frame | reconstructed BEV.

Usage: compare.py scene.json video.mp4 f1,f2,f3 out.png
Reuses heading/sample logic from bev_plot.
"""
import sys, math
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle
import matplotlib.transforms as mtransforms
sys.path.insert(0, __file__.rsplit("\\",1)[0] if "\\" in __file__ else ".")
from bev_plot import load, pos, build_tracks, sample_frame, heading, CARSZ, draw_box

def grab(video, fi):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    cap.release()
    if not ok: return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def main(scene, video, frame_list, out):
    d = load(scene); tr = build_tracks(d)
    reg = {t["track_id"]: t for t in d.get("tracks", [])}
    allx=[pos(o)[0] for seq in tr.values() for _,o in seq]
    allz=[pos(o)[1] for seq in tr.values() for _,o in seq]
    xmin,xmax=min(allx)-3,max(allx)+3
    zmin,zmax=min(allz)-3,min(max(allz)+3, max(allz)+3)
    # focus z to where action is (95th pct) for readability
    zs_sorted=sorted(allz); zfocus=zs_sorted[int(len(zs_sorted)*0.97)]+5
    import os
    zcap=float(os.environ.get("ZCAP", "0"))
    if zcap>0: zfocus=zcap
    n=len(frame_list)
    fig,axes=plt.subplots(n,2,figsize=(15,6.0*n),squeeze=False)
    for k,fi in enumerate(frame_list):
        axv=axes[k][0]; axb=axes[k][1]
        img=grab(video,fi)
        if img is not None: axv.imshow(img)
        axv.set_title(f"VIDEO frame {fi}"); axv.axis("off")
        for poly in d.get("road_polygons",[]):
            pts=[(p["x"],p["z"]) for p in poly["points"]]
            axb.add_patch(MplPoly(pts,closed=True,facecolor="#e8e8e8",edgecolor="#aaa",alpha=0.5,zorder=0))
        active=[]
        for tid in sorted(tr):
            seq=tr[tid]; frames=[f for f,_ in seq]
            if fi<frames[0] or fi>frames[-1]: continue
            p=sample_frame(seq,fi)
            if not p: continue
            cls=reg.get(tid,{}).get("class_name","")
            color=reg.get(tid,{}).get("color")
            ang=heading(seq,fi)
            draw_box(axb,p[0],p[1],ang,cls,color,tid)
            active.append(tid)
        axb.set_xlim(xmin,xmax); axb.set_ylim(zfocus,zmin)
        axb.set_aspect("equal"); axb.set_title(f"BEV frame {fi}  active={active}"); axb.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out,dpi=80); print("wrote",out)

if __name__=="__main__":
    scene,video,fl,out=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
    main(scene,video,[int(x) for x in fl.split(",")],out)
