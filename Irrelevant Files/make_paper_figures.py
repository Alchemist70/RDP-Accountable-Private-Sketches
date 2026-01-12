import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, ConnectionPatch

plt.rcParams.update({"font.size": 10})

def draw_box(ax, xy, w, h, text, facecolor="#ffd966", edgecolor="#333333", fontsize=9):
    box = FancyBboxPatch((xy[0], xy[1]), w, h, boxstyle="round,pad=0.3",
                         linewidth=1, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(box)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center", fontsize=fontsize)
    return box

def draw_arrow(ax, p0, p1, color="#0b5394", lw=2):
    ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='-|>', color=color, lw=lw))


def make_figure5(path):
    fig, ax = plt.subplots(figsize=(9,3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Boxes: Client Update -> Sketch -> Local Gaussian Noise -> Upload -> Server Aggregator -> Detector -> RDP Accounting
    boxes = []
    boxes.append(draw_box(ax, (0.2, 1.6), 1.4, 0.7, 'Client\nUpdate', facecolor='#cfe2f3'))
    boxes.append(draw_box(ax, (1.9, 1.6), 1.4, 0.7, 'Sketch\n(P\u00b7x)', facecolor='#d9ead3'))
    boxes.append(draw_box(ax, (3.6, 1.6), 1.6, 0.7, 'Local Gaussian\nPerturbation', facecolor='#f9cb9c'))
    boxes.append(draw_box(ax, (5.4, 1.6), 1.2, 0.7, 'Upload\n(or Secure\nAggregation)', facecolor='#ead1dc'))
    boxes.append(draw_box(ax, (6.7, 1.6), 1.2, 0.7, 'Server\nAggregator', facecolor='#cfe2f3'))
    boxes.append(draw_box(ax, (8.0, 1.6), 1.2, 0.7, 'Detector\n(median+MAD)', facecolor='#b4a7d6'))

    # Lower row: APS+ and RDP accounting
    boxes.append(draw_box(ax, (3.6, 0.2), 2.4, 0.7, 'APS+\n(allocates \nper-client \nsigma_i)', facecolor='#ffd966'))
    boxes.append(draw_box(ax, (6.1, 0.2), 2.6, 0.7, 'RDP\nAccounting\n(per-mechanism tuples)', facecolor='#9fc5e8'))

    # Arrows top row
    coords = [(0.2+1.4, 1.95), (1.9, 1.95),
              (1.9+1.4, 1.95), (3.6, 1.95),
              (3.6+1.6, 1.95), (5.4, 1.95),
              (5.4+1.2, 1.95), (6.7, 1.95),
              (6.7+1.2, 1.95), (8.0, 1.95)]
    draw_arrow(ax, (0.2+1.4, 1.95), (1.9,1.95))
    draw_arrow(ax, (1.9+1.4, 1.95), (3.6,1.95))
    draw_arrow(ax, (3.6+1.6, 1.95), (5.4,1.95))
    draw_arrow(ax, (5.4+1.2, 1.95), (6.7,1.95))
    draw_arrow(ax, (6.7+1.2, 1.95), (8.0,1.95))

    # Link APS+ and RDP accounting to flow
    draw_arrow(ax, (4.8, 1.6), (4.8, 0.9), color='#0b5394')
    draw_arrow(ax, (5.6, 0.9), (6.1+0.1, 0.9), color='#0b5394')
    draw_arrow(ax, (8.0+0.6, 0.9), (8.0+0.6, 1.45), color='#0b5394')

    ax.text(5.0, 2.85, 'Figure 5: PrivateSketch pipeline — client update → sketch → local noise → upload → server aggregator → detector; APS+ and RDP accounting are part of the pipeline.', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_figure6(path):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Left column: inputs
    draw_box(ax, (0.3, 1.6), 2.2, 0.9, 'Inputs:\nclient sensitivities,\nweights, global RDP target', facecolor='#d9ead3')
    # Middle: optimizer
    draw_box(ax, (3.0, 1.6), 3.6, 0.9, 'APS+ Optimizer\n(SLSQP)\nconstraint: composed RDP ≤ target', facecolor='#ffd966')
    # Right: outputs
    draw_box(ax, (6.7, 1.6), 2.6, 0.9, 'Outputs:\nper-client \nsigma_i', facecolor='#9fc5e8')

    # Arrows
    draw_arrow(ax, (0.3+2.2, 2.05), (3.0,2.05))
    draw_arrow(ax, (3.0+3.6, 2.05), (6.7,2.05))

    # Small notes and legend
    ax.text(1.4, 0.7, 'Objective: minimize weighted detection-loss surrogate (e.g., Σ w_i σ_i^2) while satisfying RDP budget', ha='center', fontsize=9)

    ax.text(5.0, 2.85, 'Figure 6: APS+ allocator flowchart — inputs → optimizer (SLSQP) → per-client σ_i outputs.', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_figure7(path):
    fig, ax = plt.subplots(figsize=(9,3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Triplet: per-mechanism tuples -> per-order RDP -> composition
    draw_box(ax, (0.4, 1.6), 2.2, 0.9, 'Per-mechanism tuples\n(q, σ, steps)', facecolor='#f9cb9c')
    draw_box(ax, (3.0, 1.6), 3.0, 0.9, 'Per-order RDP\ncompute ε_α for each α', facecolor='#cfe2f3')
    draw_box(ax, (6.5, 1.6), 2.6, 0.9, 'Compose across\nrounds & orders →\nfinal (ε, δ)', facecolor='#b4a7d6')

    draw_arrow(ax, (0.4+2.2, 2.05), (3.0, 2.05))
    draw_arrow(ax, (3.0+3.0, 2.05), (6.5, 2.05))

    ax.text(5.0, 2.85, 'Figure 7: RDP accounting pipeline — per-mechanism tuples → per-order RDP → numeric composition to (ε,δ).', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    import os
    outdir = os.path.join(os.path.dirname(__file__), '..', 'paper_figures')
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    make_figure5(os.path.join(outdir, 'pipeline_private_sketch.png'))
    make_figure6(os.path.join(outdir, 'aps_plus_flow.png'))
    make_figure7(os.path.join(outdir, 'rdp_pipeline.png'))
    print('Figures generated in', outdir)
