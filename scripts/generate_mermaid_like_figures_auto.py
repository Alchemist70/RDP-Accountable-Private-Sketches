#!/usr/bin/env python3
"""Generate Mermaid-like Figures 5-7 as vector PDFs (no emojis, exact colors).

Usage: python scripts/generate_mermaid_like_figures_auto.py
Writes PDFs and SVG previews to paper_figures/
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch
import textwrap
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'paper_figures')
os.makedirs(OUTDIR, exist_ok=True)

PALETTE = {
    'blue': '#6fa8dc',
    'green': '#93c47d',
    'orange': '#f6b26b',
    'pink': '#ead1dc',
    'teal': '#66c2a5',
    'purple': '#8e7cc3',
    'yellow': '#ffd966'
}

def draw_box(ax, xy, w, h, color, text, fontsize=9, wrap_chars=28, z=2):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', linewidth=1.4,
                         edgecolor='black', facecolor=color, zorder=z)
    ax.add_patch(box)
    # Reflow text to avoid overlapping: break into lines of ~wrap_chars
    if text:
        wrapped = '\n'.join(textwrap.fill(text, wrap_chars).splitlines())
    else:
        wrapped = ''
    ax.text(x + w/2, y + h/2, wrapped, ha='center', va='center', fontsize=fontsize)
    return (x, y, w, h)

def draw_arrow(ax, start_box, end_box, color='#222222', lw=1.2, curved=False, rad=0.0,
               start_y_frac=0.5, end_y_frac=0.5):
    """Draw arrow from right edge of start_box to left edge of end_box.

    start_box/end_box: (x,y,w,h)
    """
    x0, y0, w0, h0 = start_box
    x1, y1, w1, h1 = end_box
    p0 = (x0 + w0, y0 + h0*start_y_frac)
    p1 = (x1, y1 + h1*end_y_frac)
    if curved:
        conn_style = f"arc3,rad={rad}"
        arrow = FancyArrowPatch(p0, p1, arrowstyle=ArrowStyle('-|>', head_length=6, head_width=4),
                                linewidth=lw, color=color, shrinkA=0, shrinkB=0, connectionstyle=conn_style, zorder=4)
    else:
        arrow = FancyArrowPatch(p0, p1, arrowstyle=ArrowStyle('-|>', head_length=6, head_width=4),
                                linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=4)
    ax.add_patch(arrow)

def figure5():
    # compute total width so boxes fit cleanly in the canvas
    n = 6
    left_margin = 0.4
    right_margin = 0.6
    w = 2.3
    gap = 0.5
    total_width = left_margin + n*w + (n-1)*gap + right_margin
    # Increase canvas scale so text renders larger on-page
    fig_w, fig_h = total_width * 1.7, 3.6 * 1.7
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    # Top pipeline positions (left-to-right)
    y = 1.8
    h = 0.74
    xs = [left_margin + i*(w+gap) for i in range(n)]
    labels = [
        'Client Update',
        'Sketch\n(P·x)',
        'Local Gaussian\nPerturbation',
        'Upload (or\nSecure Aggregation)',
        'Server Aggregator',
        'Detector\n(median+MAD)'
    ]
    colors = [PALETTE['blue'], PALETTE['green'], PALETTE['orange'], PALETTE['pink'], PALETTE['teal'], PALETTE['purple']]
    boxes = []
    for x, lab, col in zip(xs, labels, colors):
        # increase font for better legibility in-PDF
        box = draw_box(ax, (x, y), w, h, col, lab, fontsize=17, wrap_chars=16, z=2)
        boxes.append(box)

    # arrows between adjacent boxes (neat, edge-to-edge)
    for a,b in zip(boxes[:-1], boxes[1:]):
        draw_arrow(ax, a, b, color='#222222', lw=1.2)

    # Supporting infra boxes placed under the Server Aggregator (avoid overlap)
    s_h = 1.0
    s_w = 3.0
    # center support inset under the Server Aggregator (boxes[4])
    s_x = xs[4] + w/2 - s_w/2
    s_y = 0.3
    draw_box(ax, (s_x, s_y), s_w, s_h, '#ffffff', '', fontsize=8, z=1)
    inner_pad = 0.14
    draw_box(ax, (s_x+inner_pad, s_y+0.22), 1.12, 0.48, PALETTE['yellow'], 'APS+\n(σᵢ allocation)', fontsize=15, wrap_chars=12, z=2)
    draw_box(ax, (s_x+inner_pad+1.32, s_y+0.22), 1.3, 0.48, PALETTE['blue'], 'RDP Acct\n(per-mech)', fontsize=15, wrap_chars=12, z=2)

    # dashed/curved connectors to support (curved paths)
    # From Local Gaussian (box index 2) down to APS+
    draw_arrow(ax, boxes[2], (s_x+inner_pad, s_y+0.22, 1.12, 0.48), color='#666666', lw=1.0, curved=False, rad=-0.3, start_y_frac=0.0, end_y_frac=0.5)
    # From Detector to RDP
    draw_arrow(ax, boxes[5], (s_x+inner_pad+2.65, s_y+0.22, 1.3, 0.48), color='#666666', lw=1.0, curved=False, rad=0.3, start_y_frac=0.0, end_y_frac=0.5)
    # No figure title text (labels removed as requested)

    out_pdf = os.path.join(OUTDIR, 'pipeline_private_sketch.pdf')
    out_svg = os.path.join(OUTDIR, 'pipeline_private_sketch.svg')
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_svg, dpi=300)
    plt.close(fig)
    print('Wrote', out_pdf, out_svg)

def figure6():
    # Increase canvas scale for better legibility when included in LaTeX
    fig_w, fig_h = 10.5 * 1.6, 1.8 * 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0, 10.5 * 1.3)
    ax.set_ylim(0, 1.8 * 1.3)
    ax.axis('off')

    y = 0.6
    w = 3.0
    h = 0.6
    xs = [0.5, 3.75, 7.0]
    labs = ['Inputs\n(client sensitivities, weights, RDP target)', 'APS+ Optimizer\n(SLSQP)\nmin Σ wᵢ σᵢ²\ns.t. composed RDP ≤ target', 'Outputs\n(per-client σᵢ)']
    cols = [PALETTE['green'], PALETTE['yellow'], PALETTE['blue']]
    boxes = []
    for x, lab, col in zip(xs, labs, cols):
        # larger font for Figure 6
        box = draw_box(ax, (x, y), w, h, col, lab, fontsize=13, wrap_chars=18)
        boxes.append(box)

    for a,b in zip(boxes[:-1], boxes[1:]):
        draw_arrow(ax, a, b, color='#222222', lw=1.2)

    # no title text (labels removed)
    out_pdf = os.path.join(OUTDIR, 'aps_plus_flow.pdf')
    out_svg = os.path.join(OUTDIR, 'aps_plus_flow.svg')
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_svg, dpi=300)
    plt.close(fig)
    print('Wrote', out_pdf, out_svg)

def figure7():
    # compute total width so boxes fit cleanly in the canvas
    n = 3
    left_margin = 0.6
    right_margin = 0.6
    w = 3.6
    gap = 0.5
    total_width = left_margin + n*w + (n-1)*gap + right_margin
    # Make Figure 7 canvas larger for in-PDF clarity
    fig_w, fig_h = total_width * 1.6, 2.2 * 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    y = 0.9
    h = 0.7
    xs = [left_margin + i*(w+gap) for i in range(n)]
    labs = ['Per-mechanism tuples\n(q, σ, steps)', 'Per-order RDP\n(compute ε_α for each α)', 'Compose across rounds & orders\n→ Final (ε, δ)']
    cols = [PALETTE['orange'], PALETTE['blue'], PALETTE['purple']]
    boxes = []
    for x, lab, col in zip(xs, labs, cols):
        # larger font for Figure 7
        box = draw_box(ax, (x, y), w, h, col, lab, fontsize=15, wrap_chars=18)
        boxes.append(box)

    for a,b in zip(boxes[:-1], boxes[1:]):
        draw_arrow(ax, a, b, color='#222222', lw=1.2)

    # no title text
    out_pdf = os.path.join(OUTDIR, 'rdp_pipeline.pdf')
    out_svg = os.path.join(OUTDIR, 'rdp_pipeline.svg')
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_svg, dpi=300)
    plt.close(fig)
    print('Wrote', out_pdf, out_svg)

def main():
    figure5()
    figure6()
    figure7()
    print('All figures generated in', OUTDIR)

if __name__ == '__main__':
    main()
