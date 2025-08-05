# -*- coding: utf-8 -*-
"""
video_freq.py  – Find dominant vibration frequency in a processed video
---------------------------------------------------------------
usage examples
--------------
python video_freq.py --video duct_mag.avi
python video_freq.py --video duct_mag.avi --roi 80,120,160,140 --step 2
"""

from __future__ import print_function
import argparse, sys, os
import logging
import time
import numpy as np
from scipy.signal import detrend, windows, find_peaks
try:
    # Py-3 (scipy.fft)
    from scipy.fft import rfft, rfftfreq
except ImportError:
    # Py-2 fallback
    from scipy.fftpack import rfft, rfftfreq

try:
    import cv2                                    # OpenCV ≥ 3.0
except ImportError:
    sys.exit("ERROR: OpenCV (cv2) is not installed.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--video", required=True, help="input video (already magnified)")
p.add_argument("--roi",  help="x,y,w,h (pixels). Default: full frame")
p.add_argument("--step", type=int, default=1, help="use every N-th frame (speed)")
p.add_argument("--plot", action="store_true", help="plot spectrum if matplotlib found")
args = p.parse_args()

# log start
start_wall = time.time()
#start_cpu = time.process_time()
logging.info("Start frequency analysis on %s" %args.video)

# ---------- open video ----------
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    sys.exit("Cannot open video file: %s" % args.video)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ret, frame = cap.read()
if not ret:
    sys.exit("Video contains no frames.")
h, w = frame.shape[:2]

# ROI defaults to whole frame
x, y, ww, hh = 0, 0, w, h
if args.roi:
    x, y, ww, hh = [int(v) for v in args.roi.split(",")]

# ---------- collect trace ----------
trace = []
frame_idx = 0
while ret:
    if frame_idx % args.step == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+hh, x:x+ww]
        trace.append(roi.mean())
    ret, frame = cap.read()
    frame_idx += 1
cap.release()

if len(trace) < 4:
    sys.exit("Too few samples; try smaller --step or longer video.")

trace = np.asarray(trace, dtype=np.float64)

# ---------- FFT ----------
sig = detrend(trace - trace.mean())
N   = len(sig)
win = windows.hann(N)
mag = np.abs(rfft(sig * win))
freqs = rfftfreq(N, d=1.0/fps*args.step)

lowcut = 5.0
highcut = 25.0

valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
if valid_idx.size == 0:
    sys.exit("No FFT bins in the range %s-%s Hz!" % (lowcut, highcut))

# extract magnitude from band    
band_mag = mag[valid_idx]
band_freq = freqs[valid_idx]

# find all local maxima in that band
peaks_idx, _ = find_peaks(band_mag, height=0)

# sort those peaks by descending magnitude
order = np.argsort(band_mag[peaks_idx])[::-1]
top_order = order[:5] # top 5 peaks
top_peaks = peaks_idx[top_order]

peak_list = []
df = freqs[1] - freqs[0]

for idx in top_peaks:
    k = valid_idx[idx]
    a, b, c = mag[k-1], mag[k], mag[k+1] if k+1 < len(mag) else (0, mag[k], 0)
    delta = 0.5 * (a - c) / (a - 2*b + c) if (a -2*b + c ) != 0 else 0.0
    f_est = (k + delta) * df
    peak_list.append((f_est, mag[k]))


# end logging
end_wall = time.time()
#end_cpu = time.process_time()
elapsed = end_wall - start_wall
logging.info("Total wall-clock time: %.2fs" % (elapsed))

print("\nTop 5 frequency in %.1f-%.1f  Hz   (fps = %.1f, frames used = %d)"
      % (lowcut, highcut, fps/args.step, N))
for i, (fval, mval) in enumerate(peak_list, 1):
    print("  %d. %6.2f Hz  |  % .2e" % (i, fval, mval))

# ---------- optional plot ----------
if args.plot:
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(1, 1, 1)
        ax.semilogy(freqs, mag, linewidth=1)
        #plt.title("Peak = %.2f Hz" % peak_list[0])
        ax.set_xlabel("Frequency (Hz)") 
        ax.set_ylabel("|FFT|")
        ax.set_xlim(lowcut, highcut)
        ax.grid(True, linestyle=":")

        # anotate each peak
        for fval, mval in peak_list:
            ax.plot([fval], [mval], 'ro')
            ax.text(fval, mval*1.2, "%.2fHz" % (fval), color='red', fontsize=8, ha='center')

        # add a small table of the top 5
        cell_text = [[ "%.2f" % f, "%.2e" % m ] for f,m in peak_list]
        table = ax.table(
         cellText=cell_text,
         colLabels=["Freq (Hz)", "Mag"],
         cellLoc="center",
         loc="upper right"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1,1.5)
 
        ax.set_title("Top 5 peaks in %.1f-%.1f Hz" % (lowcut, highcut))
        plt.tight_layout()
        png_name = os.path.splitext(args.video)[0] + "_spectrum.png"
        plt.savefig(png_name, dpi=150)
        print("Spectrum saved to:", png_name)
    except ImportError:
        print("(matplotlib not installed → skipping plot)")
