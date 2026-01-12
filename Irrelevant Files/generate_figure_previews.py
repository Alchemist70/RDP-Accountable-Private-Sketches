#!/usr/bin/env python3
"""
Generate PNG previews of the three PDF figures for quick visual verification.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def extract_pdf_to_image(pdf_path, output_path, dpi=150):
    """Extract first page of PDF and save as PNG."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        pix.save(output_path)
        return True
    except:
        pass
    
    # Fallback: use ghostscript if available
    try:
        import subprocess
        subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER',
                       f'-r{dpi}', '-sDEVICE=pngalpha',
                       f'-sOutputFile={output_path}', pdf_path],
                       check=True, capture_output=True)
        return True
    except:
        pass
    
    return False


if __name__ == '__main__':
    paper_figs_dir = os.path.join(os.path.dirname(__file__), '..', 'paper_figures')
    
    figures = [
        'pipeline_private_sketch.pdf',
        'aps_plus_flow.pdf', 
        'rdp_pipeline.pdf'
    ]
    
    for fig_name in figures:
        pdf_path = os.path.join(paper_figs_dir, fig_name)
        png_path = os.path.join(paper_figs_dir, fig_name.replace('.pdf', '_preview.png'))
        
        if os.path.exists(pdf_path):
            if extract_pdf_to_image(pdf_path, png_path, dpi=150):
                print(f'✓ Generated {png_path}')
            else:
                print(f'✗ Could not convert {pdf_path}')
        else:
            print(f'✗ File not found: {pdf_path}')
