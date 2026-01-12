"""make_paper_figures_gpt.py

Generate Figures 5,6,7 as deterministic vector PDFs from a JSON spec.

Run:
    python scripts/make_paper_figures_gpt.py

Requires:
    python 3.8+, numpy, matplotlib

This script reads `figure_spec.json` in the repository root and writes
the PDFs to the `paper_figures/` directory. Overlays are written as
`*_overlay.tex` files suitable for use with the `overpic` package.
"""
import os
import json
import textwrap
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path


def load_spec():
    here = os.path.dirname(os.path.dirname(__file__))
    spec_path = os.path.join(here, 'figure_spec.json')
    with open(spec_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def wrap_label(s, width):
    return "\n".join(textwrap.fill(line, width) for line in s.splitlines())


def draw_band(ax, x, w, color, label=None, fontsize=8, align='center', y=0.08, h=0.84,
              draw_text=True, labels_list=None, axes_bbox=(0.0, 0.15, 1.0, 0.7)):
    rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='k', linewidth=0.6)
    ax.add_patch(rect)
    if label and draw_text:
        if align == 'left':
            ax.text(x + 0.03 * w, y + h / 2, label, ha='left', va='center', fontsize=fontsize)
        else:
            ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fontsize)
    elif label and (not draw_text) and labels_list is not None:
        axes_left, axes_bottom, axes_width, axes_height = axes_bbox
        if align == 'left':
            lx = axes_left + (x + 0.03 * w) * axes_width
        else:
            lx = axes_left + (x + 0.5 * w) * axes_width
        ly = axes_bottom + (y + h / 2.0) * axes_height
        labels_list.append({'text': label, 'x_pct': lx * 100.0, 'y_pct': ly * 100.0, 'align': align, 'fontsize': fontsize})


def bezier_between(x0, x1, y0=0.4, y1=0.6, curviness=0.22):
    dx = x1 - x0
    c1 = (x0 + dx * curviness, y0)
    c2 = (x1 - dx * curviness, y1)
    verts = [(x0, y0), c1, c2, (x1, y1)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


def draw_crossing_curves(ax, centers, ncurves=6, stroke=1.6, color='#111111'):
    for i in range(len(centers) - 1):
        x0 = centers[i]
        x1 = centers[i + 1]
        for k in range(ncurves):
            t = (k + 1) / (ncurves + 1)
            y0 = 0.46 - 0.12 * (t - 0.5)
            y1 = 0.54 - 0.12 * (0.5 - t)
            path = bezier_between(x0, x1, y0=y0, y1=y1, curviness=0.22)
            patch = PathPatch(path, facecolor='none', edgecolor=color, lw=stroke, alpha=0.95)
            ax.add_patch(patch)


def latex_math_label_lines(raw_label: str):
    """Sanitize a raw label and return a list of LaTeX math-mode lines.

    This removes/escapes problematic characters, replaces known symbols,
    expands greek names, converts subscripts, and wraps each line in $...$.
    """
    def sanitize(s):
        # Replace Unicode bullets and centered-dots with asterisk
        s = s.replace('\u2022', '*').replace('\u00b7', '*').replace('·', '*')
        s = s.replace('\\textperiodcentered', '\\cdot')
        # Keep only ASCII 32-126 (printable ASCII) plus newline; escape LaTeX specials
        result = []
        for c in s:
            if c == '\n':
                result.append('\n')
            elif 32 <= ord(c) < 127:
                # Escape LaTeX special characters
                if c == '&':
                    result.append('\\&')
                elif c == '#':
                    result.append('\\#')
                elif c == '%':
                    result.append('\\%')
                elif c == '$':
                    result.append('\\$')
                elif c == '_':
                    result.append('_')  # Keep underscores; we'll handle subscripts later
                elif c == '{':
                    result.append('\\{')
                elif c == '}':
                    result.append('\\}')
                elif c == '\\':
                    result.append('\\textbackslash{}')
                elif c == '^':
                    result.append('\\textasciicircum{}')
                elif c == '~':
                    result.append('\\textasciitilde{}')
                else:
                    result.append(c)
        return ''.join(result)

    lab = sanitize(raw_label)
    parts = [p.strip() for p in lab.split('\n') if p.strip()]
    greek_map = {
        'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma', 'delta': '\\delta',
        'epsilon': '\\epsilon', 'sigma': '\\sigma', 'pi': '\\pi', 'mu': '\\mu',
        'rho': '\\rho', 'omega': '\\omega', 'phi': '\\phi',
    }
    out_lines = []
    for p in parts:
        for k, v in greek_map.items():
            p = re.sub(rf'(?<!\\)\\?{k}', lambda m: v, p)
        p = re.sub(r'([a-zA-Z0-9])_([a-zA-Z0-9]+)', r'\1_{\2}', p)
        out_lines.append(f'${p}$')
    return out_lines


def make_figure(spec_item, outdir):
    w_in, h_in = spec_item.get('canvas_inches', [11.0, 2.2])
    bands = spec_item['bands']
    bottom = spec_item.get('bottom_bands', None)
    curves = spec_item.get('crossing_curves', None)
    fontsize = spec_item.get('label_fontsize', 8)
    wrap_width = spec_item.get('label_wrap', 36)

    fig = plt.figure(figsize=(w_in, h_in))
    axes_bbox_main = (0.0, 0.15, 1.0, 0.7)
    ax = fig.add_axes(list(axes_bbox_main))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fracs = np.array([b.get('width_frac', 1.0) for b in bands], dtype=float)
    fracs = fracs / fracs.sum()
    xpos = np.concatenate(([0.0], np.cumsum(fracs[:-1])))

    labels_for_overlay = []
    centers = []
    for i, b in enumerate(bands):
        x = float(xpos[i])
        w = float(fracs[i])
        draw_band(ax, x, w, b.get('color', '#cccccc'), label=wrap_label(b.get('label', ''), wrap_width),
                  fontsize=fontsize, align=b.get('label_align', 'center'), draw_text=False,
                  labels_list=labels_for_overlay, axes_bbox=axes_bbox_main)
        centers.append(x + w / 2.0)

    if bottom:
        axes_bbox_bottom = (0.0, 0.03, 1.0, 0.10)
        b_fracs = np.array([b.get('width_frac', 1.0) for b in bottom], dtype=float)
        b_fracs = b_fracs / b_fracs.sum()
        b_xpos = np.concatenate(([0.0], np.cumsum(b_fracs[:-1])))
        for i, bb in enumerate(bottom):
            bx = float(b_xpos[i])
            bw = float(b_fracs[i])
            draw_band(ax, bx, bw, bb.get('color', '#dddddd'), label=wrap_label(bb.get('label', ''), wrap_width),
                      fontsize=fontsize, align=bb.get('label_align', 'center'), draw_text=False,
                      labels_list=labels_for_overlay, axes_bbox=axes_bbox_bottom)

    if curves and centers:
        n_between = curves.get('n_between', 6)
        stroke = curves.get('stroke', 1.6)
        color = curves.get('color', '#111111')
        draw_crossing_curves(ax, centers, ncurves=n_between, stroke=stroke, color=color)

    outpath = spec_item.get('output_filename', os.path.join(outdir, 'figure.pdf'))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches='tight')
    # Also write a high-resolution PNG preview to aid visual inspection
    try:
        png_path = outpath.replace('.pdf', '.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    except Exception:
        # if PNG writing fails (missing backend), continue silently
        pass

    overlay_path = outpath.replace('.pdf', '_overlay.tex')
    with open(overlay_path, 'w', encoding='utf-8') as f:
        f.write('% Overlay labels for {}\n'.format(os.path.basename(outpath)))
        f.write('% Generated by make_paper_figures_gpt.py — use with overpic package.\n')
        for lbl in labels_for_overlay:
            raw = lbl['text']
            x_pct = round(lbl['x_pct'], 2)
            y_pct = round(lbl['y_pct'], 2)
            anchor = 'l' if lbl.get('align', 'left') == 'left' else 'c'
            lines = latex_math_label_lines(raw)
            # smaller spacing (percent points) so multi-line labels stack tightly
            line_spacing = 1.6
            for i, line in enumerate(lines):
                y_offset = y_pct - (len(lines) - 1) * line_spacing / 2.0 + i * line_spacing
                f.write('\\put({:.2f},{:.2f}){{\\makebox(0,0)[{}]{{{}}}}}\n'.format(x_pct, y_offset, anchor, line))

    plt.close(fig)
    return outpath


def main():
    spec = load_spec()
    repo_root = os.path.dirname(os.path.dirname(__file__))
    outdir = os.path.join(repo_root, 'paper_figures')
    generated = []
    for key in ['fig5', 'fig6', 'fig7']:
        if key in spec:
            path = make_figure(spec[key], outdir)
            generated.append(path)
            print('Generated', path)
    print('Done. Generated {} figures.'.format(len(generated)))


if __name__ == '__main__':
    main()
 
