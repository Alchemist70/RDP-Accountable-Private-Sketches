#!/usr/bin/env python3
"""
Generate clean, professional Figures 5-7 for the PrivateSketch paper.
- No figure labels/titles embedded in images
- Clean, minimal design
- Professional appearance
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import rcParams

# Professional figure settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.5
rcParams['lines.linewidth'] = 1.0
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


# Color palette
COLORS = {
    'blue': '#5B8DBE',
    'green': '#70AD47',
    'orange': '#E89B3C',
    'pink': '#D8A8CC',
    'teal': '#5FB3BC',
    'purple': '#8E7CC3',
    'yellow': '#FFD966',
}


def draw_box(ax, x, y, w, h, color, label, fontsize=7.5):
    """Draw a colored box with label."""
    rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='#555555', 
                     linewidth=1.0, alpha=0.9)
    ax.add_patch(rect)
    
    if label:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
               fontsize=fontsize, fontweight='normal', wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='#333333', width=1.5):
    """Draw an arrow."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle='-|>', mutation_scale=18, 
                           linewidth=width, color=color, zorder=10)
    ax.add_patch(arrow)


def draw_bezier_curves(ax, centers, n_curves=5, y_base=0.5, amplitude=0.08, 
                      color='#444444', linewidth=0.8):
    """Draw smooth Bezier curves between centers."""
    for i in range(len(centers)-1):
        x0 = centers[i]
        x1 = centers[i+1]
        for k in range(n_curves):
            t = (k + 1) / (n_curves + 1)
            y0 = y_base - amplitude * (t - 0.5)
            y1 = y_base + amplitude * (0.5 - t)
            
            dx = x1 - x0
            c1_x = x0 + dx * 0.3
            c1_y = y0
            c2_x = x1 - dx * 0.3
            c2_y = y1
            
            verts = [(x0, y0), (c1_x, c1_y), (c2_x, c2_y), (x1, y1)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor='none', edgecolor=color, 
                            linewidth=linewidth, alpha=0.6)
            ax.add_patch(patch)


def make_figure_5(output_dir):
    """Figure 5: PrivateSketch pipeline."""
    fig, ax = plt.subplots(figsize=(11.5, 2.0), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Top row: 6 pipeline stages
    stages = [
        ('Client\nUpdate', COLORS['blue'], 0.12),
        ('Sketch\n(P·x)', COLORS['green'], 0.13),
        ('Local Gaussian\nNoise', COLORS['orange'], 0.15),
        ('Upload/Secure\nAggregation', COLORS['pink'], 0.15),
        ('Server\nAggregator', COLORS['teal'], 0.14),
        ('Detector\n(median+MAD)', COLORS['purple'], 0.15),
    ]
    
    x_pos = 0.01
    y_main = 0.50
    h_band = 0.40
    centers = []
    
    for label, color, width in stages:
        draw_box(ax, x_pos, y_main, width, h_band, color, label, fontsize=7)
        centers.append(x_pos + width/2)
        x_pos += width + 0.005
    
    # Draw data flow curves
    draw_bezier_curves(ax, centers, n_curves=4, y_base=0.70, amplitude=0.08, 
                      color='#555555', linewidth=0.9)
    
    # Bottom row: APS+ and RDP
    y_bottom = 0.05
    h_bottom = 0.35
    
    draw_box(ax, 0.27, y_bottom, 0.35, h_bottom, COLORS['yellow'],
            'APS+\n(allocates σᵢ)', fontsize=7)
    draw_box(ax, 0.63, y_bottom, 0.35, h_bottom, COLORS['blue'],
            'RDP Accounting\n(per-mechanism tuples)', fontsize=7)
    
    # Connection lines
    draw_arrow(ax, 0.445, y_main, 0.445, 0.42, color='#666666', width=0.9)
    draw_arrow(ax, 0.81, y_main, 0.81, 0.42, color='#666666', width=0.9)
    
    fig.tight_layout(pad=0.02)
    outfile = os.path.join(output_dir, 'pipeline_private_sketch.pdf')
    fig.savefig(outfile, format='pdf', bbox_inches='tight', dpi=300)
    print(f'✓ Figure 5: {outfile}')
    plt.close(fig)


def make_figure_6(output_dir):
    """Figure 6: APS+ allocator."""
    fig, ax = plt.subplots(figsize=(10.5, 1.5), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Three sections
    sections = [
        ('Inputs:\nClient sensitivities\nweights, RDP target',
         COLORS['green'], 0.28),
        ('APS+ Optimizer\n(SLSQP)\nmin Σ wᵢ σᵢ² s.t. RDP ≤ target',
         COLORS['yellow'], 0.44),
        ('Outputs:\nPer-client σᵢ',
         COLORS['blue'], 0.22),
    ]
    
    x_pos = 0.01
    y_main = 0.25
    h_band = 0.65
    centers = []
    
    for label, color, width in sections:
        draw_box(ax, x_pos, y_main, width, h_band, color, label, fontsize=6.5)
        centers.append(x_pos + width/2)
        x_pos += width + 0.03
    
    # Arrows
    draw_arrow(ax, centers[0] + 0.14, y_main + h_band/2,
              centers[1] - 0.22, y_main + h_band/2, color='#333333', width=1.2)
    draw_arrow(ax, centers[1] + 0.22, y_main + h_band/2,
              centers[2] - 0.11, y_main + h_band/2, color='#333333', width=1.2)
    
    fig.tight_layout(pad=0.02)
    outfile = os.path.join(output_dir, 'aps_plus_flow.pdf')
    fig.savefig(outfile, format='pdf', bbox_inches='tight', dpi=300)
    print(f'✓ Figure 6: {outfile}')
    plt.close(fig)


def make_figure_7(output_dir):
    """Figure 7: RDP accounting."""
    fig, ax = plt.subplots(figsize=(11.5, 1.5), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Three stages
    stages = [
        ('Per-mechanism tuples\n(q, σ, steps)',
         COLORS['orange'], 0.26),
        ('Per-order RDP\nCompute εₐ for each α',
         COLORS['blue'], 0.40),
        ('Compose across\nrounds & orders\n→ (ε, δ)',
         COLORS['purple'], 0.26),
    ]
    
    x_pos = 0.01
    y_main = 0.25
    h_band = 0.65
    centers = []
    
    for label, color, width in stages:
        draw_box(ax, x_pos, y_main, width, h_band, color, label, fontsize=7)
        centers.append(x_pos + width/2)
        x_pos += width + 0.04
    
    # Arrows
    draw_arrow(ax, centers[0] + 0.13, y_main + h_band/2,
              centers[1] - 0.20, y_main + h_band/2, color='#333333', width=1.2)
    draw_arrow(ax, centers[1] + 0.20, y_main + h_band/2,
              centers[2] - 0.13, y_main + h_band/2, color='#333333', width=1.2)
    
    fig.tight_layout(pad=0.02)
    outfile = os.path.join(output_dir, 'rdp_pipeline.pdf')
    fig.savefig(outfile, format='pdf', bbox_inches='tight', dpi=300)
    print(f'✓ Figure 7: {outfile}')
    plt.close(fig)


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(repo_root, 'paper_figures')
    os.makedirs(output_dir, exist_ok=True)
    
    print('='*70)
    print('Generating Clean, Professional Figures 5-7')
    print('='*70)
    
    make_figure_5(output_dir)
    make_figure_6(output_dir)
    make_figure_7(output_dir)
    
    print('='*70)
    print('✅ All figures generated successfully')
    print('='*70)


if __name__ == '__main__':
    main()
