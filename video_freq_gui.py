# -*- coding: utf-8 -*-
"""
video_freq_gui.py  –  pick ROI with the mouse, then auto-estimate frequency
---------------------------------------------------------------------------
Run:
    python video_freq_gui.py --video duct_mag.avi [--step 2] [--plot]
Mouse:
    •  left-button drag  = draw ROI
Keyboard:
    •  ENTER/RETURN      = accept ROI and start FFT
    •  r                 = reset / redraw ROI
    •  q or ESC          = quit without processing
"""

from __future__ import print_function
import argparse, cv2, numpy as np, sys, os
from scipy.signal import detrend, windows
try:
    from scipy.fft import rfft, rfftfreq          # Py-3
except ImportError:
    from scipy.fftpack import rfft, rfftfreq      # Py-2

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument('--video', required=True)
ap.add_argument('--step', type=int, default=1, help='use every N-th frame')
ap.add_argument('--plot', action='store_true')
args = ap.parse_args()

# ---------- open video ----------
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    sys.exit('Cannot open video: %s' % args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ret, first = cap.read()
if not ret:
    sys.exit('Video empty')

clone = first.copy()
roi_box = None            # (x0,y0,x1,y1)
drawing = False

def on_mouse(event, x, y, flags, _):
    global roi_box, drawing, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_box = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_box[2], roi_box[3] = x, y
        img = clone.copy()
        cv2.rectangle(img, (roi_box[0], roi_box[1]), (x, y), (0,255,0), 2)
        cv2.imshow('select ROI', img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_box[2], roi_box[3] = x, y
        if roi_box[2] < roi_box[0]: roi_box[0],roi_box[2] = roi_box[2],roi_box[0]
        if roi_box[3] < roi_box[1]: roi_box[1],roi_box[3] = roi_box[3],roi_box[1]
        img = clone.copy()
        cv2.rectangle(img,(roi_box[0],roi_box[1]),(roi_box[2],roi_box[3]),(0,255,0),2)
        cv2.imshow('select ROI', img)

cv2.namedWindow('select ROI')
cv2.setMouseCallback('select ROI', on_mouse)
cv2.imshow('select ROI', first)

print('Draw ROI, press ENTER to accept, r = reset, q = quit')
while True:
    key = cv2.waitKey(0) & 0xFF
    if key in (13, 10):   # ENTER
        if roi_box is None:
            print('No ROI selected!'); continue
        break
    elif key in (ord('r'), ord('R')):
        clone = first.copy()
        roi_box = None
        cv2.imshow('select ROI', clone)
    elif key in (ord('q'), 27):
        sys.exit('Cancelled by user')

cv2.destroyAllWindows()
x0,y0,x1,y1 = roi_box
w,h = x1-x0, y1-y0
print('ROI: (%d,%d) %dx%d' % (x0,y0,w,h))

# ---------- build intensity trace ----------
trace = []
frame_idx = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # rewind
while True:
    ret, fr = cap.read()
    if not ret: break
    if frame_idx % args.step == 0:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        trace.append(gray[y0:y1, x0:x1].mean())
    frame_idx += 1
cap.release()

trace = np.asarray(trace)
if len(trace) < 4:
    sys.exit('Too few frames for FFT')

# ---------- FFT ----------
sig  = detrend(trace - trace.mean())
N    = len(sig)
win  = windows.hann(N)
mag  = np.abs(rfft(sig * win))
freq = rfftfreq(N, d=1.0/(fps/args.step))
peak = freq[1:][np.argmax(mag[1:])]

print('\nDominant frequency :  %.2f Hz   (fps=%.1f  frames=%d)'
      % (peak, fps/args.step, N))

# ---------- optional plot ----------
if args.plot:
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3))
        plt.semilogy(freq, mag)
        plt.title('Peak = %.2f Hz' % peak)
        plt.xlabel('Hz'); plt.ylabel('|FFT|')
        plt.tight_layout(); plt.grid(True, ls=':')
        out_png = os.path.splitext(args.video)[0] + '_spectrum.png'
        plt.savefig(out_png, dpi=150)
        print('Spectrum saved to:', out_png)
    except ImportError:
        print('(matplotlib not available → skipping plot)')
