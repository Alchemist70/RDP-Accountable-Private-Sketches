import re
infile = 'paper_acm_draft.ocr.tex'
outfile = 'paper_acm_draft.ocr.sanit.tex'

with open(infile, 'r', encoding='utf-8', errors='replace') as f:
    s = f.read()
# Common mojibake replacements
replacements = {
    'Â«': '<<',
    'Â»': '>>',
    'â€”': '---',
    'â€“': '--',
    'â€™': "'",
    'â€œ': '"',
    'â€\x9d': '"',
    'â€': '"',
    '\x0c': '',
    '\x9d': '',
}
for a,b in replacements.items():
    s = s.replace(a,b)
# Remove other control chars except newline and tab
s = ''.join(ch for ch in s if (ch=='\n' or ch=='\t' or ord(ch)>=32))

# Ensure preamble contains T1 font encoding and lmodern
s = s.replace('\\usepackage[utf8]{inputenc}', '\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}')

with open(outfile, 'w', encoding='utf-8') as f:
    f.write(s)
print('Wrote sanitized file', outfile)
