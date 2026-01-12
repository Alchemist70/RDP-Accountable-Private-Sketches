from PyPDF2 import PdfReader
import sys
try:
    from PyPDF2 import PdfReader
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyPDF2'])
    from PyPDF2 import PdfReader

pdf_path = 'paper_acm_draft.pdf'
reader = PdfReader(pdf_path)
print(f'Pages: {len(reader.pages)}')
for i, page in enumerate(reader.pages, start=1):
    print(f'--- Page {i} ---')
    resources = page.get('/Resources')
    if resources is None:
        print(' No resources')
        continue
    if hasattr(resources, 'get_object'):
        resources = resources.get_object()
    xobj = resources.get('/XObject')
    if xobj is None:
        print(' No XObject')
        continue
    if hasattr(xobj, 'get_object'):
        xobj = xobj.get_object()
    try:
        items = list(xobj.keys())
    except Exception:
        items = []
    imgs = []
    for name in items:
        obj = xobj.get(name)
        try:
            if hasattr(obj, 'get_object'):
                obj = obj.get_object()
            subtype = obj.get('/Subtype')
        except Exception:
            subtype = None
        if subtype == '/Image':
            imgs.append(name)
    print(f' Image XObjects: {len(imgs)}')
    for n in imgs:
        print('  ', n)
