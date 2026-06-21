"""Montage labeled video frames into one image. Usage: montage.py out.png img1 img2 ..."""
import sys, cv2, numpy as np
out = sys.argv[1]; paths = sys.argv[2:]
imgs = []
for p in paths:
    im = cv2.imread(p)
    label = p.split("/")[-1].split("\\")[-1]
    cv2.putText(im, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    imgs.append(im)
# normalize widths
W = max(im.shape[1] for im in imgs)
imgs = [cv2.resize(im, (W, int(im.shape[0]*W/im.shape[1]))) for im in imgs]
# 2 columns
rows = []
for i in range(0, len(imgs), 2):
    pair = imgs[i:i+2]
    if len(pair) == 1:
        pair.append(np.zeros_like(pair[0]))
    rows.append(np.hstack(pair))
grid = np.vstack(rows)
cv2.imwrite(out, grid)
print("wrote", out, grid.shape)
