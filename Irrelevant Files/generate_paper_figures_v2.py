#!/usr/bin/env python3
"""
Generate Figures 5-7 for the PrivateSketch paper.

Figures:
  - Figure 5: PrivateSketch pipeline (client → sketch → noise → server → detector)
  - Figure 6: APS+ allocator flowchart (inputs → optimizer → outputs)
  - Figure 7: RDP accounting pipeline (per-mechanism tuples → per-order RDP → final (ε,δ))

This script creates professional, publication-ready PDF figures with:
  - Clean band/section layout with proper proportions
  - Clear labels and data flow indicators
  - Accurate color palette and typography
  - Vector-based output (PDF) for publication
"""

import os
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import rcParams

# Professional figure settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 0.5
rcParams['lines.linewidth'] = 1.5
rcParams['pdf.fonttype'] = 42  # Use TrueType fonts in PDF
rcParams['ps.fonttype'] = 42


# Color palette: professional, distinct, publication-ready
PALETTE = {
    'client_update': '#6fa8dc',      # Blue
    'sketch': '#93c47d',              # Green
    'noise': '#f6b26b',               # Orange
    'upload': '#ead1dc',              # Pink
    'aggregator': '#66c2a5',          # Teal
    'detector': '#8e7cc3',            # Purple
    'aps_plus': '#ffd966',            # Golden yellow
    'rdp': '#6fa8dc',                 # Blue
    'inputs': '#93c47d',              # Green
    'optimizer': '#ffd966',           # Golden yellow
    'outputs': '#6fa8dc',             # Blue
    'per_mechanism': '#f6b26b',       # Orange
    'per_order': '#6fa8dc',           # Blue
    'compose': '#8e7cc3',             # Purple
}


def wrap_text(text, width=16):
    """Wrap text to specified width."""
    return '\n'.join(textwrap.fill(line, width) for line in text.split('\n'))


def draw_band(ax, x, y, w, h, color, label, fontsize=8, align='center'):
    """Draw a colored band with centered/left-aligned label."""
    rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='#333333', 
                     linewidth=0.8, alpha=0.95)
    ax.add_patch(rect)
    
    if label:
        if align == 'left':
            ax.text(x + 0.02*w, y + h/2, label, ha='left', va='center', 
                   fontsize=fontsize, fontweight='normal', wrap=True)
        elif align == 'center':
            ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                   fontsize=fontsize, fontweight='normal', wrap=True)
        else:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                   fontsize=fontsize, fontweight='normal', wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='#333333', width=1.2, style='-|>'):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle=style, mutation_scale=15, 
                           linewidth=width, color=color, zorder=10)
    ax.add_patch(arrow)


def draw_bezier_curves(ax, x_centers, n_curves=6, y_base=0.5, amplitude=0.1, 
                      color='#111111', linewidth=1.2):
    """Draw Bezier curves connecting band centers showing data flow."""
    for i in range(len(x_centers)-1):
        x0 = x_centers[i]
        x1 = x_centers[i+1]
        for k in range(n_curves):
            t = (k + 1) / (n_curves + 1)
            y0 = y_base - amplitude * (t - 0.5)
            y1 = y_base + amplitude * (0.5 - t)
            
            # Create Bezier curve control points
            dx = x1 - x0
            c1_x = x0 + dx * 0.3
            c1_y = y0
            c2_x = x1 - dx * 0.3
            c2_y = y1
            
            verts = [(x0, y0), (c1_x, c1_y), (c2_x, c2_y), (x1, y1)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor='none', edgecolor=color, 
                            linewidth=linewidth, alpha=0.7)
            ax.add_patch(patch)


def make_figure_5(output_dir):
    """Generate Figure 5: PrivateSketch pipeline."""
    fig, ax = plt.subplots(figsize=(11, 2.4), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Main pipeline bands (top row)
    stages = [
        ('Client\nUpdate', PALETTE['client_update'], 0.10),
        ('Sketch\n(P·x)', PALETTE['sketch'], 0.14),
        ('Local Gaussian\nPerturbation', PALETTE['noise'], 0.16),
        ('Upload\n(or Secure\nAggregation)', PALETTE['upload'], 0.14),
        ('Server\nAggregator', PALETTE['aggregator'], 0.14),
        ('Detector\n(median+MAD)', PALETTE['detector'], 0.14),
    ]
    
    x_pos = 0.02
    y_main = 0.45
    h_band = 0.40
    centers = []
    
    for label, color, width in stages:
        draw_band(ax, x_pos, y_main, width, h_band, color, label, fontsize=7.5, align='center')
        centers.append(x_pos + width/2)
        x_pos += width + 0.01
    
    # Draw data flow curves
    draw_bezier_curves(ax, centers, n_curves=5, y_base=0.65, amplitude=0.08, 
                      color='#222222', linewidth=1.0)
    
    # Supporting infrastructure (bottom row)
    y_support = 0.02
    h_support = 0.30
    
    # APS+ box (left support)
    draw_band(ax, 0.32, y_support, 0.32, h_support, PALETTE['aps_plus'], 
             'APS+\n(allocates per-client σᵢ)', fontsize=7, align='center')
    
    # RDP Accounting box (right support)
    draw_band(ax, 0.66, y_support, 0.32, h_support, PALETTE['rdp'],
             'RDP Accounting\n(per-mechanism tuples)', fontsize=7, align='center')
    
    # Connection lines from pipeline to support
    draw_arrow(ax, 0.48, y_main, 0.48, 0.38, color='#0b5394', width=1.0)
    draw_arrow(ax, 0.82, y_main, 0.82, 0.38, color='#0b5394', width=1.0)
    
    # Title
    ax.text(0.5, 0.95, 'Figure 5: PrivateSketch Pipeline',
           ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.91, 
           'Client update → Sketch → Local noise → Server aggregator → Detector; APS+ allocates noise & RDP Accounting records tuples',
           ha='center', va='top', fontsize=7.5, style='italic')
    
    fig.tight_layout(pad=0.05)
    outfile = os.path.join(output_dir, 'figure_5_pipeline.pdf')
    fig.savefig(outfile, format='pdf', bbox_inches='tight', dpi=300)
    print(f'✓ Generated {outfile}')
    plt.close(fig)
    
    return outfile


def make_figure_6(output_dir):
    """Generate Figure 6: APS+ allocator flowchart."""
    fig, ax = plt.subplots(figsize=(10.5, 1.8), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Three main sections: Inputs → Optimizer → Outputs
    stages = [
        ('Inputs:\nClient sensitivities,\nweights, global RDP target',
         PALETTE['inputs'], 0.24),
        ('APS+ Optimizer (SLSQP)\nConstraint: composed RDP ≤ target\nObjective: minimize Σ wᵢ σᵢ²',
         PALETTE['optimizer'], 0.50),
        ('Outputs:\nPer-client σᵢ',
         PALETTE['outputs'], 0.20),
    ]
    
    x_pos = 0.02
    y_main = 0.35
    h_band = 0.50
    centers = []
    
    for label, color, width in stages:
        draw_band(ax, x_pos, y_main, width, h_band, color, label, fontsize=7.5, align='center')
        centers.append(x_pos + width/2)
        x_pos += width + 0.02
    
    # Draw data flow arrows
    draw_arrow(ax, centers[0] + 0.12, y_main + h_band/2, 
              centers[1] - 0.25, y_main + h_band/2, color='#333333', width=1.2)
    draw_arrow(ax, centers[1] + 0.25, y_main + h_band/2,
              centers[2] - 0.10, y_main + h_band/2, color='#333333', width=1.2)
    
    # Title
    ax.text(0.5, 0.95, 'Figure 6: APS+ Allocator Flowchart',
           ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.90,
           'Inputs → SLSQP Optimization → Per-client Noise Allocations',
           ha='center', va='top', fontsize=7.5, style='italic')
    
    fig.tight_layout(pad=0.05)
    outfile = os.path.join(output_dir, 'figure_6_aps_plus.pdf')
    fig.savefig(outfile, format='pdf', bbox_inches='tight', dpi=300)
    print(f'✓ Generated {outfile}')
    plt.close(fig)
    
    return outfile


def make_figure_7(output_dir):
    """Generate Figure 7: RDP accounting pipeline."""
    fig, ax = plt.subplots(figsize=(11, 1.8), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Three-stage RDP composition pipeline
    stages = [
        ('Per-mechanism tuples\n(q, σ, steps)',
         PALETTE['per_mechanism'], 0.26),
        ('Per-order RDP\nCompute εₐ for each α',
         PALETTE['per_order'], 0.38),
        ('Compose across\nrounds & orders\n→ final (ε, δ)',
         PALETTE['compose'], 0.28),
    ]
    
    x_pos = 0.02
    y_main = 0.35
    h_band = 0.50
    centers = []
    
    for label, color, width in stages:
        draw_band(ax, x_pos, y_main, width, h_band, color, label, fontsize=7.5, align='center')
        centers.append(x_pos + width/2)
        x_pos += width + 0.02
    
    # Draw data flow arrows
    draw_arrow(ax, centers[0] + 0.13, y_main + h_band/2,
              centers[1] - 0.19, y_main + h_band/2, color='#333333', width=1.2)
    draw_arrow(ax, centers[1] + 0.19, y_main + h_band/2,
              centers[2] - 0.14, y_main + h_band/2, color='#333333', width=1.2)
    
    # Title
    ax.text(0.5, 0.95, 'Figure 7: RDP Accounting Pipeline',
           ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.90,
           'Per-mechanism tuples → Per-order RDP computation → Numeric composition → Final privacy bound',
           ha='center', va='top', fontsize=7.5, style='italic')
    
    fig.tight_layout(pad=0.05)
    outfile = os.path.join(output_dir, 'figure_7_rdp.pdf')
    fig.savefig(outfile, format='pdf', bbox_inches='tight', dpi=300)
    print(f'✓ Generated {outfile}')
    plt.close(fig)
    
    return outfile


def main():
    """Generate all three figures."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(repo_root, 'paper_figures')
    os.makedirs(output_dir, exist_ok=True)
    
    print('='*70)
    print('Generating Publication-Ready Figures 5-7 for PrivateSketch Paper')
    print('='*70)
    
    # Generate all three figures
    fig5 = make_figure_5(output_dir)
    fig6 = make_figure_6(output_dir)
    fig7 = make_figure_7(output_dir)
    
    print('='*70)
    print(f'All figures generated successfully in: {output_dir}')
    print('='*70)
    
    return fig5, fig6, fig7


if __name__ == '__main__':
    main()
