import sys
p = r'c:\Users\rravi\FL_Improvements_Research\scripts\make_paper_figures_gpt.py'
with open(p, 'rb') as f:
    for i, line in enumerate(f, start=1):
        if i<=60:
            print(i, line)
        else:
            break
