# -*- coding: utf-8 -*-
"""
vreq.py  – Find and filter the true mechanical vibration frequency
---------------------------------------------------------------
usage:
    python vreq.py --video duct_mag.avi
    python vreq.py --video duct_mag.avi --roi x,y,w,h --step 2 --plot
"""

from __future__ import print_function
import argparse, sys, os, logging, time
import numpy as np
from scipy.signal import detrend, windows, find_peaks, butter, filtfilt
try:
    from scipy.fft import rfft, rfftfreq
except ImportError:
    from scipy.fftpack import rfft, rfftfreq

# OpenCV
try:
    import cv2
except ImportError:
    sys.exit("ERROR: OpenCV (cv2) is not installed.")

# -- logging setup --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--video", required=True, help="input video (already magnified)")
p.add_argument("--roi",  help="x,y,w,h (pixels). Default: full frame")
p.add_argument("--step", type=int, default=1, help="use every N-th frame")
p.add_argument("--plot", action="store_true", help="plot spectra and tables")
args = p.parse_args()

# log start
start_wall = time.time()
logging.info("Start frequency analysis on %s" % args.video)

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
    try:
        x, y, ww, hh = [int(v) for v in args.roi.split(",")]
    except:
        sys.exit("Invalid --roi format, expected x,y,w,h")

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

# ---------- FFT helper ----------
def compute_fft(data, fps, step):
    sig   = detrend(data - data.mean())
    N     = len(sig)
    win   = windows.hann(N)
    fft_r = np.fft.rfft(sig * win)
    mag   = np.abs(fft_r)
    # sample‐spacing is step/fps seconds
    freqs = np.fft.rfftfreq(N, d=float(step)/fps)
    return freqs, mag

# ---------- 1) Raw vs Diff FFT ----------
freqs_raw, mag_raw = compute_fft(trace, fps, args.step)
idx_raw = np.argmax(mag_raw[1:]) + 1
peak_raw = freqs_raw[idx_raw]
print("Raw-trace FFT peak at %.2f Hz" % peak_raw)

diff_trace = np.diff(trace)
freqs_diff, mag_diff = compute_fft(diff_trace, fps, args.step)

# --- NEW: restrict search to physical band (e.g. 5–25 Hz) ---
phys_low, phys_high = 5.0, 25.0
band_mask = (freqs_diff >= phys_low) & (freqs_diff <= phys_high)
if not np.any(band_mask):
    sys.exit("No diff-trace FFT bins inside the physical band %.1f–%.1f Hz" %
             (phys_low, phys_high))

# zero out bins outside the band and find peak inside
masked_mag = mag_diff.copy()
masked_mag[~band_mask] = 0.0
idx_diff = np.argmax(masked_mag[1:]) + 1
peak_diff = freqs_diff[idx_diff]
print("Diff-trace FFT peak at %.2f Hz (mechanical freq)" % peak_diff)

# ---------- 2) Band-pass filter around that peak ----------
fs  = fps / float(args.step)
nyq = 0.5 * fs

f0 = peak_diff
lowcut  = max(phys_low, f0 - 1.0)
highcut = min(phys_high, f0 + 1.0, nyq - 1e-3)  # clamp below Nyquist

if lowcut >= highcut:
    print("Degenerate band-pass window (lowcut>=highcut); skipping filter.")
    filtered = diff_trace.copy()
else:
    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype="band")
    filtered = filtfilt(b, a, diff_trace)

# FFT of filtered
freqs_filt, mag_filt = compute_fft(filtered, fps, args.step)
idx_filt  = np.argmax(mag_filt[1:]) + 1
peak_filt = freqs_filt[idx_filt]
print("\nAfter bandpass %.2f-%.2f Hz -> FFT peak = %.2f Hz" %
      (lowcut, highcut, peak_filt))
#print("Filtered freqs min/max:", freqs_filt.min(), freqs_filt.max())
#print("  First few filtered freqs:", freqs_filt[:10])


# ---------- 3) Top-5 peaks in filtered spectrum ----------
freqs, mag = freqs_filt, mag_filt
valid_idx  = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
if valid_idx.size == 0:
    sys.exit("No FFT bins in the range %s-%s Hz!" % (lowcut, highcut))

band_mag   = mag[valid_idx]
peaks_idx, _ = find_peaks(band_mag, height=0)
order = np.argsort(band_mag[peaks_idx])[::-1]
top_order = order[:5]
top_peaks = peaks_idx[top_order]

df = freqs[1] - freqs[0]
peak_list = []
for idx in top_peaks:
    k = valid_idx[idx]
    a = mag[k-1] if k>0 else 0
    b = mag[k]
    c = mag[k+1] if k+1 < len(mag) else 0
    delta = 0.5*(a - c)/(a - 2*b + c) if (a - 2*b + c)!=0 else 0
    f_est = (k + delta)*df
    peak_list.append((f_est, mag[k]))

end_wall = time.time()
elapsed  = end_wall - start_wall
logging.info("Total wall-clock time: %.2fs" % elapsed)

print("\nTop 5 frequencies in %.1f-%.1f Hz  (fps=%.1f, samples=%d)" %
      (lowcut, highcut, fps/args.step, len(filtered)))
for i,(fval,mval) in enumerate(peak_list,1):
    print("  %d. %6.2f Hz  |  % .2e" % (i,fval,mval))

# ---------- 4) plotting (optional) ----------
if args.plot:
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14,4))

        # panel 1: raw
        ax1 = fig.add_subplot(1,3,1)
        ax1.semilogy(freqs_raw, mag_raw, linewidth=1)
        ax1.axvline(peak_raw, color='r', linestyle='--')
        ax1.set_xlim(0, highcut*2)
        ax1.set_title("Raw-trace FFT\npeak=%.2f Hz" % peak_raw)
        ax1.set_xlabel("Hz")
        ax1.grid(True)

        # panel 2: diff
        ax2 = fig.add_subplot(1,3,2)
        ax2.semilogy(freqs_diff, mag_diff, linewidth=1)
        ax2.axvline(peak_diff, color='r', linestyle='--')
        ax2.set_xlim(0, highcut*2)
        ax2.set_title("Diff-trace FFT\npeak=%.2f Hz" % peak_diff)
        ax2.set_xlabel("Hz")
        ax2.grid(True)

        # panel 3: filtered + top5
        ax3 = fig.add_subplot(1,3,3)
        ax3.semilogy(freqs, mag, linewidth=1)
        ax3.set_xlim(lowcut-1, highcut+1)
        ax3.set_title("Filtered FFT\npeak=%.2f Hz" % peak_filt)
        ax3.set_xlabel("Hz")
        ax3.grid(True)
        # overlay top5 markers
        for fval,mval in peak_list:
            ax3.plot(fval, mval, 'ro', markersize=6)
        # table
        cell_text = [[ "%.2f" % f, "%.2e" % m ] for f,m in peak_list]
        tbl = ax3.table(
            cellText=cell_text,
            colLabels=["Freq (Hz)","Mag"],
            cellLoc="center",
            loc="lower right"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1,1.5)

        plt.tight_layout()
        out_png = os.path.splitext(args.video)[0] + "_filtered_spectrum.png"
        plt.savefig(out_png, dpi=150)
        print("Saved plot to:", out_png)

    except ImportError:
        print("(matplotlib not installed → skipping plot)")
