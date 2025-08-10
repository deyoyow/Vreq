
# vreq — Video Frequency Estimation Using Fast Fourier Transform

**vreq** estimates the **dominant vibration frequency** from a **motion-magnified video** (produced by a Phase-Based Motion Magnification, PBMM, pipeline).  
It provides a simple, non-contact alternative to conventional sensor-based vibrometry for structures like HVAC ducting, beams, and plates.

- **Input**: a PBMM output video (e.g., 4K/60fps, magnified in a temporal band such as 5–25 Hz).  
- **Processing**: ROI reduction → detrending & windowing → **FFT / Welch PSD** → **peak picking** (optional plot/export).  
- **Output**: dominant frequency estimate (Hz), and optional spectrum visualization.

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Usage](#command-line-usage)
- [How It Works](#how-it-works)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)

---

## Repository Structure

- [vreq.py] # Frequency estimation CLI (use this to analyze PBMM videos)     
- [phasebasedMoMag.py] # (Optional) PBMM demo script to create    motion-magnified videos   
- [pyramid2arr.py] # Helper to convert pyramid    coeffs to arrays (PBMM side)   
- [media/Sample/] # Example videos / put your test clips here
- [README.md]

> **Note:** `vreq.py` is the main tool for **estimating frequency from an already magnified video**.  
> `phasebasedMoMag.py` is included only if you also want to **generate** a PBMM video locally.

---

## Requirements

### For frequency estimation (`vreq.py`)
- Python **3.9+** (tested on 3.10/3.11)
- `numpy`, `scipy`, `opencv-python`, `matplotlib`

### For creating PBMM videos (optional)
- Complex steerable pyramid implementation, e.g.:
  ```bash
  pip install perceptual
-   A separate environment is recommended if you rely on legacy PBMM scripts.
    

> The original PBMM demo code that inspired this repo was historically written for **Python 2.7**.  
> This repository’s **estimator** works on Python 3. Keep PBMM generation in its own env if needed.

----------

## Installation

Create a virtual environment (recommended) and install dependencies.

**Windows (PowerShell):**
```bash
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install numpy scipy opencv-python matplotlib
# Optional (PBMM generation)
pip install perceptual
```
## Quick Start

1.  Put a **PBMM video** in `media/Sample/`, e.g.:
    

```powershell
`media/Sample/1.mp4-Mag20Ideal-lo5-hi25.avi` 
```
2.  Run the estimator (show a plot and print the dominant frequency):
    
```powershell
`python vreq.py --video media/Sample/1.mp4-Mag20Ideal-lo5-hi25.avi --plot` 
```
3.  (Optional) Save the spectrum plot to file:
```powershell
`python vreq.py --video media/Sample/1.mp4-Mag20Ideal-lo5-hi25.avi --plot --save-fig spectrum.png` 
```
> Tip: run `python vreq.py -h` to see all available options in your version.
## Command-Line Usage

Common flags (exact set may vary by your script version):

```bash

 --video <path> Path to the PBMM video to analyze (required)
 --roi <x,y,w,h> ROI in pixels; defaults to heuristics or full frame  
 --method <fft|welch>  Spectrum method (default: welch)  
 --fs <fps>  Override frame rate if metadata is missing/wrong 
 --plot Show spectrum figure  
 --save-fig <path> Save spectrum figure (PNG/PDF)  
 --out <path> Save text/JSON summary with dominant frequency 
```
Examples:

```bash

# Welch PSD, automatic ROI, show plot 
python vreq.py --video media/Sample/1.mp4-Mag20Ideal-lo5-hi25.avi --plot 
# Specify ROI and save a figure 
python vreq.py --video media/Sample/1.mp4-Mag20Ideal-lo5-hi25.avi \ --roi 320,180,640,360 --plot --save-fig result.png 
```
----------

## How It Works

1.  **Frame Reduction**: Each frame is reduced to a **time series** by averaging intensities in a **Region of Interest (ROI)** (or another reduction strategy you choose).
    
2.  **Preprocessing**: The resulting trace is **detrended** and **windowed** (e.g., Hann) to mitigate DC drift and spectral leakage.
    
3.  **Spectrum**: Compute amplitude spectrum via **FFT** or **Welch PSD** and perform **parabolic peak interpolation** to refine the dominant frequency estimate.
    
4.  **Output**: Print or save the dominant frequency (Hz) and, if requested, display or export the spectrum plot.
    

----------

## Best Practices

-   **Frame Rate & Nyquist**: The highest unaliased frequency is `fps/2` (Nyquist). Choose PBMM’s temporal band and your target frequency accordingly.
    
-   **PBMM Band**: If you generate PBMM videos, set the **temporal band-pass** around your expected vibration band (e.g., 5–25 Hz).
    
-   **ROI Selection**: Align the ROI with the vibrating structure; avoid mixing background that does not move coherently.
    
-   **Stability**: Use a tripod; avoid camera shake. Rolling shutter and lighting flicker (50/60 Hz and harmonics) can bias the spectrum.
    
-   **Window Length**: Longer windows → finer frequency resolution; shorter windows → lower latency. Match to your application.
    
-   **Consistency**: If you compare amplitudes across experiments, keep preprocessing identical (detrend, window type, normalization).
    

----------

## Troubleshooting

-   **“Could not open video”**  
    Check the path and file permissions; verify codecs (`opencv-python` can read most MP4/AVI with system codecs).
    
-   **Dominant peak looks wrong / too low**  
    Confirm the video **FPS** is correct; ensure your target motion is below **Nyquist** (`fps/2`); extend the time window; refine ROI.
    
-   **No clear peak**  
    Try a larger ROI over the most active area, increase PBMM gain (when generating the input), or use **Welch** with more averaging.
    
-   **Git push rejected (remote has updates)**  
    Run:
    
   ``` bash
    
    `git pull --rebase origin main
     git push -u origin main` 
   ```

----------

## Acknowledgements

The PBMM reference code in this repository was implemented from jvgemert/pbMoMa while the Fast Fourier Transform is an independent work, the code for PBMM may contain modification/changes that makes it differ from the original repository.

---
## License
---
## Citation





