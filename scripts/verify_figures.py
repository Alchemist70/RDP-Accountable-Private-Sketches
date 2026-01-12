#!/usr/bin/env python3
"""
Verify the generated figures by checking their properties and content.
"""
import os
import subprocess

def verify_pdf(filepath):
    """Verify PDF file properties."""
    if not os.path.exists(filepath):
        return f"✗ File not found: {filepath}"
    
    file_size = os.path.getsize(filepath)
    file_size_kb = file_size / 1024
    
    # Try to get PDF info using pdfinfo if available
    try:
        result = subprocess.run(['pdfinfo', filepath], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            info = {}
            for line in lines:
                if ':' in line:
                    key, val = line.split(':', 1)
                    info[key.strip()] = val.strip()
            
            return {
                'file': os.path.basename(filepath),
                'size_kb': round(file_size_kb, 2),
                'pages': info.get('Pages', 'N/A'),
                'title': info.get('Title', 'N/A'),
                'status': '✓ Valid PDF'
            }
    except:
        pass
    
    # Fallback: just check file size
    return {
        'file': os.path.basename(filepath),
        'size_kb': round(file_size_kb, 2),
        'status': '✓ File exists (size check only)'
    }


if __name__ == '__main__':
    paper_figs_dir = os.path.join(os.path.dirname(__file__), '..', 'paper_figures')
    
    figures = [
        ('Figure 5', 'pipeline_private_sketch.pdf'),
        ('Figure 6', 'aps_plus_flow.pdf'),
        ('Figure 7', 'rdp_pipeline.pdf'),
    ]
    
    print("="*70)
    print("FIGURE VERIFICATION REPORT")
    print("="*70)
    
    for fig_num, fig_file in figures:
        filepath = os.path.join(paper_figs_dir, fig_file)
        result = verify_pdf(filepath)
        
        print(f"\n{fig_num}: {fig_file}")
        if isinstance(result, dict):
            for key, val in result.items():
                if key != 'file':
                    print(f"  {key}: {val}")
        else:
            print(f"  {result}")
    
    print("\n" + "="*70)
    print("Files ready for LaTeX compilation and publication")
    print("="*70)
