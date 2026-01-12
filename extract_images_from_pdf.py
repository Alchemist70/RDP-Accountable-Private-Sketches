from PyPDF2 import PdfReader
import sys
from pathlib import Path

pdf_path = 'paper_acm_draft.pdf'
outdir = Path('extracted_images')
outdir.mkdir(exist_ok=True)
reader = PdfReader(pdf_path)
for i, page in enumerate(reader.pages, start=1):
    resources = page.get('/Resources')
    if resources is None:
        continue
    if hasattr(resources, 'get_object'):
        resources = resources.get_object()
    xobj = resources.get('/XObject')
    if xobj is None:
        continue
    if hasattr(xobj, 'get_object'):
        xobj = xobj.get_object()
    for name in list(xobj.keys()):
        obj = xobj.get(name)
        if hasattr(obj, 'get_object'):
            obj = obj.get_object()
        subtype = obj.get('/Subtype')
        if subtype == '/Image':
            data = obj.get_data()
            filter_ = obj.get('/Filter')
            ext = '.bin'
            if filter_ == '/DCTDecode':
                ext = '.jpg'
            elif filter_ == '/JPXDecode':
                ext = '.jp2'
            elif filter_ == '/FlateDecode':
                ext = '.png'
            keyname = str(name)
            keyname = keyname.lstrip('/\\')
            fname = outdir / f'page{i}_{keyname}{ext}'
            with open(str(fname), 'wb') as f:
                f.write(data)
            print(f'Wrote {fname} (filter={filter_})')
