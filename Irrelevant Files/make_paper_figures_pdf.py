import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})

PALETTE = {
    'blue': '#6fa8dc',
    'green': '#93c47d',
    'orange': '#f6b26b',
    'pink': '#ead1dc',
    'purple': '#8e7cc3',
    'yellow': '#ffd966',
    'teal': '#66c2a5'
}


def draw_band(ax, x, w, color, label=None, text_kwargs=None, y=0.08, h=0.84, align='center'):
    # draw band with thin black border for clearer separation
    rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='k', linewidth=0.6)
    ax.add_patch(rect)
    if label:
        tk = {'ha': 'center', 'va': 'center', 'fontsize': 8, 'color': 'k'}
        if text_kwargs:
            tk.update(text_kwargs)
        if align == 'left':
            ax.text(x + 0.03 * w, y + h/2, label, ha='left', va='center', fontsize=tk.get('fontsize', 8))
        else:
            ax.text(x + w/2, y + h/2, label, **tk)


def bezier_between(x0, x1, y0=0.28, y1=0.72, curviness=0.22):
    # deterministic cubic bezier control points between (x0,y0) and (x1,y1)
    dx = x1 - x0
    c1 = (x0 + dx * curviness, y0)
    c2 = (x1 - dx * curviness, y1)
    verts = [
        (x0, y0),
        c1,
        c2,
        (x1, y1)
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


def draw_crossing_curves(ax, xpos, widths, ncurves=6, color='#111111'):
    # deterministic crossing curves between band centers
    centers = [x + w/2 for x, w in zip(xpos, widths)]
    for i in range(len(centers) - 1):
        x0 = centers[i]
        x1 = centers[i+1]
        # spread curves vertically across a narrow band to mimic the attachment
        for k in range(ncurves):
            t = (k + 1) / (ncurves + 1)
            # y positions interpolate between slightly below center to slightly above
            y0 = 0.45 - 0.18 * (t - 0.5)
            y1 = 0.55 - 0.18 * (0.5 - t)
            path = bezier_between(x0, x1, y0=y0, y1=y1, curviness=0.22)
            patch = PathPatch(path, facecolor='none', edgecolor=color, lw=1.6, alpha=0.95)
            ax.add_patch(patch)


def make_pipeline_fig(outdir):
    fig_w = 11.0
    fig_h = 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    names = ['Client Update', 'Sketch (P·x)', 'Local Gaussian\nPerturbation',
             'Upload (or Secure)\nAggregation', 'Server Aggregator', 'Detector\n(median+MAD)']
    cols = [PALETTE['blue'], PALETTE['green'], PALETTE['orange'], PALETTE['pink'], PALETTE['teal'], PALETTE['purple']]
    # proportional widths
    widths = np.array([0.14, 0.16, 0.18, 0.16, 0.18, 0.18])
    widths = widths / widths.sum()
    xpos = np.concatenate(([0.0], np.cumsum(widths[:-1])))
    for x, w, c, n in zip(xpos, widths, cols, names):
        draw_band(ax, x, w, c, label=n, text_kwargs={'fontsize': 7})

    # bottom long bands for APS+ and RDP
    bottom_y = -0.12
    # draw a second row by placing a new axes overlay for the bottom row
    ax2 = fig.add_axes([0.0, 0.04, 1.0, 0.28])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    b_names = ['APS+ (allocates per-client σ_i)', 'RDP Accounting (per-mechanism tuples)']
    b_cols = [PALETTE['yellow'], PALETTE['blue']]
    b_widths = np.array([0.48, 0.52])
    b_xpos = np.concatenate(([0.0], np.cumsum(b_widths[:-1])))
    for x, w, c, n in zip(b_xpos, b_widths, b_cols, b_names):
        draw_band(ax2, x, w, c, label=n, text_kwargs={'fontsize': 7})

    # crossing curves on main row
    draw_crossing_curves(ax, xpos, widths, ncurves=6, color='#222222')

    fig.savefig(os.path.join(outdir, 'pipeline_private_sketch.pdf'), bbox_inches='tight')
    plt.close(fig)


def make_aps_fig(outdir):
    fig_w = 10.5
    fig_h = 1.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    names = ['Inputs:\nclient sensitivities, weights,\nglobal RDP target', 'APS+ Optimizer\n(SLSQP)\nconstraint: composed RDP ≤ target', 'Outputs:\nper-client σ_i']
    cols = [PALETTE['green'], PALETTE['yellow'], PALETTE['blue']]
    widths = np.array([0.33, 0.34, 0.33])
    xpos = np.concatenate(([0.0], np.cumsum(widths[:-1])))
    for x, w, c, n in zip(xpos, widths, cols, names):
        draw_band(ax, x, w, c, label=n, text_kwargs={'fontsize': 8})

    draw_crossing_curves(ax, xpos, widths, ncurves=5, color='#222222')
    fig.savefig(os.path.join(outdir, 'aps_plus_flow.pdf'), bbox_inches='tight')
    plt.close(fig)


def make_rdp_fig(outdir):
    fig_w = 11.0
    fig_h = 1.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    names = ['Per-mechanism tuples\n(q, σ, steps)', 'Per-order RDP\ncompute ε_α for each α', 'Compose across\nrounds & orders → final (ε, δ)']
    cols = [PALETTE['orange'], PALETTE['blue'], PALETTE['purple']]
    widths = np.array([0.30, 0.40, 0.30])
    xpos = np.concatenate(([0.0], np.cumsum(widths[:-1])))
    for x, w, c, n in zip(xpos, widths, cols, names):
        draw_band(ax, x, w, c, label=n, text_kwargs={'fontsize': 8})

    draw_crossing_curves(ax, xpos, widths, ncurves=6, color='#222222')

    fig.savefig(os.path.join(outdir, 'rdp_pipeline.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(__file__), '..', 'paper_figures')
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    make_pipeline_fig(outdir)
    make_aps_fig(outdir)
    make_rdp_fig(outdir)
    print('PDF figures generated in', outdir)
