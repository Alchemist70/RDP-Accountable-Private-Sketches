import os
from pdf2image import convert_from_path
import pytesseract

PDF='Paper.pdf'
OUTTXT='Paper_ocr.txt'
OUTTEX='paper_acm_draft.ocr.tex'

# Set tesseract binary path explicitly
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print('Converting PDF to images...')
images = convert_from_path(PDF, dpi=300)
print(f'Got {len(images)} pages')

all_text = []
for i, img in enumerate(images, start=1):
    print(f'OCR page {i}/{len(images)}')
    text = pytesseract.image_to_string(img, lang='eng')
    all_text.append(text)

full_text = '\n\n'.join(all_text)
with open(OUTTXT, 'w', encoding='utf-8') as f:
    f.write(full_text)
print(f'Wrote OCR text to {OUTTXT} ({len(full_text)} chars)')

# Heuristic section splitting
headings = ['Abstract','Introduction','Related Work','Background','Method','Methods','Approach','Experiments','Results','Discussion','Conclusion','Conclusions','References','Appendix']
sections = {}
current = 'FrontMatter'
sections[current] = []
for line in full_text.splitlines():
    sline = line.strip()
    if not sline:
        sections[current].append('')
        continue
    # detect heading exact match (case-insensitive)
    for h in headings:
        if sline.lower().startswith(h.lower()) and len(sline.split())<=5:
            current = h
            sections[current] = []
            # if heading line contains extra text like '1 Introduction', skip the heading text
            break
    else:
        sections[current].append(sline)

# Compose a simple LaTeX document
def escape_tex(s):
    return s.replace('\\','\\textbackslash{}').replace('%','\\%').replace('&','\\&').replace('#','\\#').replace('_','\\_').replace('{','\\{').replace('}','\\}')

with open(OUTTEX, 'w', encoding='utf-8') as f:
    f.write('% Auto-generated LaTeX from OCR (may need manual cleanup)\n')
    f.write('\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{hyperref}\n\\begin{document}\n')
    # write Abstract specially if present
    if 'Abstract' in sections:
        f.write('\\begin{abstract}\n')
        f.write(escape_tex('\n'.join(sections['Abstract']).strip()) + '\n')
        f.write('\\end{abstract}\n')
    # write other sections in order if present
    order = ['Introduction','Background','Method','Methods','Approach','Experiments','Results','Discussion','Conclusion','Conclusions','Related Work','References','Appendix']
    for o in order:
        if o in sections:
            f.write('\\section*{' + o + '}\n')
            f.write(escape_tex('\n'.join(sections[o]).strip()) + '\n\n')
    # fallback: front matter
    if sections.get('FrontMatter'):
        f.write('\\section*{FrontMatter}\n')
        f.write(escape_tex('\n'.join(sections['FrontMatter']).strip()) + '\n')
    f.write('\\end{document}\n')

print(f'Wrote basic LaTeX to {OUTTEX}')
