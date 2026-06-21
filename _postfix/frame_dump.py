"""Save individual full-res video frames as PNGs for ground-truth inspection.
Usage: frame_dump.py video.mp4 f1,f2,... prefix
"""
import sys, cv2
video, fl, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total frames:", total)
for f in [int(x) for x in fl.split(",")]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    if not ok:
        print("fail", f); continue
    out = f"{prefix}_{f}.png"
    cv2.imwrite(out, img)
    print("wrote", out, img.shape)
cap.release()
