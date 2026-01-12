#!/usr/bin/env python3
"""
Overlay smooth Bézier data-flow curves on fig_5.png and save edited image.
Saves: fig_5_edited.png (root) and paper_figures/pipeline_private_sketch_withflow.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PIL import Image

infile = os.path.join(os.path.dirname(__file__), '..', 'fig_5.png')
outroot = os.path.join(os.path.dirname(__file__), '..')
out1 = os.path.join(outroot, 'fig_5_edited.png')
out2 = os.path.join(outroot, 'paper_figures', 'pipeline_private_sketch_withflow.png')

if not os.path.exists(infile):
    print('Input file not found:', infile)
    raise SystemExit(1)

offsets = [-6, -2, 2, 6]  # pixel offsets vertical
img = Image.open(infile).convert('RGBA')
W, H = img.size

# Try to detect top row of content (non-white) to place curves more robustly
arr = np.array(img)
gray = arr[..., :3].mean(axis=2)
rows_nonwhite = np.where(gray.mean(axis=1) < 250)[0]
if len(rows_nonwhite) > 0:
    y_min = int(rows_nonwhite.min())
    y_max = int(rows_nonwhite.max())
else:
    y_min = int(0.10 * H)
    y_max = int(0.30 * H)

# Heuristic top row center line
y_top = y_min + int((y_max - y_min) * 0.35)

# Estimate horizontal centers by finding peaks in column color variance
cols_var = gray.var(axis=0)
from scipy.signal import find_peaks
peaks, _ = find_peaks(cols_var, distance= int(W/8))
if len(peaks) >= 6:
    centers_x = peaks[:6].tolist()
else:
    margin = int(0.05 * W)
    available = W - 2*margin
    n = 6
    centers_x = [margin + (i + 0.5) * (available / n) for i in range(n)]

# Setup matplotlib figure with background image
fig = plt.figure(figsize=(W/100, H/100), dpi=100)
ax = fig.add_axes([0,0,1,1])
ax.imshow(img)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis('off')

# Cleaner curve parameters
color = '#444444'
curve_count = 3
offsets = [-12, 0, 12]
main_lw = 1.6
sec_lw = 1.0
main_alpha = 0.75
sec_alpha = 0.45

# Draw curves (behind boxes) — add slightly above the box centerline
for i in range(len(centers_x)-1):
    x0 = float(centers_x[i])
    x1 = float(centers_x[i+1])
    dx = x1 - x0
    # endpoints placed slightly above box center
    y0 = y_top + int(0.12 * H)
    y1 = y0
    for k, off in enumerate(offsets):
        oy = off
        c1 = (x0 + dx * 0.28, y0 - oy)
        c2 = (x1 - dx * 0.28, y1 - oy)
        verts = [(x0, y0), c1, c2, (x1, y1)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        if k == 1:
            lw = main_lw
            alpha = main_alpha
        else:
            lw = sec_lw
            alpha = sec_alpha
        patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, alpha=alpha, capstyle='round', joinstyle='round')
        ax.add_patch(patch)
        # arrowhead on middle curve only
        if k == 1:
            ax.annotate('', xy=(x1-8, y1), xytext=(x1-28, y1), arrowprops=dict(arrowstyle='-|>', color=color, lw=lw))

# Save outputs: PNG (preview) and vector SVG+PDF for LaTeX
os.makedirs(os.path.join(outroot, 'paper_figures'), exist_ok=True)
fig.savefig(out1, dpi=300, bbox_inches='tight', pad_inches=0)
svg_out = out2.replace('.png', '.svg')
pdf_out = out2.replace('.png', '.pdf')
fig.savefig(svg_out, bbox_inches='tight', pad_inches=0)
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print('Saved edited images:', out1, svg_out, pdf_out)
