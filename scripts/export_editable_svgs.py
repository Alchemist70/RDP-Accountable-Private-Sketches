import os
import base64
from PIL import Image
import numpy as np
import io
from xml.sax.saxutils import escape

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

def detect_centers(img_arr, W, H):
    gray = img_arr[..., :3].mean(axis=2)
    if find_peaks is not None:
        cols_var = gray.var(axis=0)
        distance = max(10, int(W/8))
        peaks, _ = find_peaks(cols_var, distance=distance)
        if len(peaks) >= 6:
            return peaks[:6].tolist(), int(H*0.18)
    # fallback evenly spaced
    margin = int(0.05 * W)
    available = W - 2*margin
    n = 6
    centers_x = [int(margin + (i + 0.5) * (available / n)) for i in range(n)]
    y_top = int(0.18 * H)
    return centers_x, y_top

def make_svg(outfile, img_path, centers_x, y_top, W, H):
    # embed image as base64 PNG
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('ascii')

    svg_parts = []
    svg_parts.append(f'<?xml version="1.0" encoding="UTF-8"?>')
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    svg_parts.append(f'<defs><style><![CDATA[')
    svg_parts.append('.bezier { fill: none; stroke: #444444; stroke-linecap: round; stroke-linejoin: round; }')
    svg_parts.append('.ctrl { fill: #ff3333; stroke: #000000; stroke-width:0.6 }')
    svg_parts.append('.label { font-family: Arial, sans-serif; font-size:12px; fill:#111111 }')
    svg_parts.append(']]></style></defs>')
    svg_parts.append(f'<image href="data:image/png;base64,{img_b64}" x="0" y="0" width="{W}" height="{H}" preserveAspectRatio="none" />')

    offsets = [-12, 0, 12]
    # draw paths and control points
    path_id = 1
    for i in range(len(centers_x)-1):
        x0 = float(centers_x[i])
        x1 = float(centers_x[i+1])
        dx = x1 - x0
        y0 = y_top + int(0.12 * H)
        y1 = y0
        for k, off in enumerate(offsets):
            oy = off
            c1x = x0 + dx * 0.28
            c1y = y0 - oy
            c2x = x1 - dx * 0.28
            c2y = y1 - oy
            strokew = 1.6 if k==1 else 1.0
            alpha = 0.85 if k==1 else 0.55
            svg_parts.append(f'<path id="p{path_id}" class="bezier" d="M {x0} {y0} C {c1x} {c1y}, {c2x} {c2y}, {x1} {y1}" stroke-width="{strokew}" opacity="{alpha}" />')
            # control points
            svg_parts.append(f'<circle cx="{c1x}" cy="{c1y}" r="4" class="ctrl" />')
            svg_parts.append(f'<text x="{c1x+6}" y="{c1y-6}" class="label">p{path_id}.c1</text>')
            svg_parts.append(f'<circle cx="{c2x}" cy="{c2y}" r="4" class="ctrl" />')
            svg_parts.append(f'<text x="{c2x+6}" y="{c2y-6}" class="label">p{path_id}.c2</text>')
            path_id += 1

    # also add invisible rectangle for easier selection in Inkscape
    svg_parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="none" stroke="none" />')
    svg_parts.append('</svg>')

    with open(outfile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_parts))

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(root, 'fig_5.png')
    outdir = os.path.join(root, 'paper_figures')
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(img_path):
        print('Error: expected image', img_path)
        return
    img = Image.open(img_path).convert('RGBA')
    W, H = img.size
    arr = np.array(img)
    centers_x, y_top = detect_centers(arr, W, H)

    out_svg = os.path.join(outdir, 'pipeline_private_sketch_withflow_editable.svg')
    make_svg(out_svg, img_path, centers_x, y_top, W, H)

    # Make a copy for Excalidraw import
    out_svg_ex = os.path.join(outdir, 'pipeline_private_sketch_withflow_for_excalidraw.svg')
    with open(out_svg, 'rb') as r, open(out_svg_ex, 'wb') as w:
        w.write(r.read())

    # Write a short README
    readme = os.path.join(outdir, 'README_import_edit.md')
    with open(readme, 'w', encoding='utf-8') as f:
        f.write('# Import instructions\n')
        f.write('* `pipeline_private_sketch_withflow_editable.svg`: Inkscape-ready SVG. Open in Inkscape and drag control point circles to edit curve handles.\n')
        f.write('* `pipeline_private_sketch_withflow_for_excalidraw.svg`: Import this SVG into Excalidraw (File → Import) for further editing. Excalidraw will convert SVG elements into editable shapes.\n')

    print('Wrote:', out_svg, out_svg_ex, readme)

if __name__ == '__main__':
    main()
