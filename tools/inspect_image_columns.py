from PIL import Image, ImageOps
import sys
import numpy as np

path = sys.argv[1]
img = Image.open(path).convert('L')
# Resize to manageable width if very large
w,h = img.size
if w>2000:
    img = img.resize((2000, int(2000*h/w)), Image.LANCZOS)
arr = np.array(img)
# Normalize: detect background as near-white
th = np.percentile(arr, 98)
mask = arr < th  # True for non-background
col_sum = mask.sum(axis=0)
# Smooth with small window
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
smooth = gaussian_filter1d(col_sum.astype(float), sigma=3)
# Find peaks
peaks, props = find_peaks(smooth, height=max(1, smooth.max()*0.15), distance=5)
print('width,height:', img.size)
print('detected_peaks:', len(peaks))
print('peak_positions:', list(peaks))
# Print some stats
print('col_sum_max:', int(col_sum.max()), 'median:', int(np.median(col_sum)))
