from setuptools import setup, find_packages

setup(
    name="pbmoma",
    version="0.1",
    packages=find_packages(where="."),  # assumes all .py files are in the top‐level directory
    py_modules=["phase_based_motion_magnification", "pyr2arr", "temporal_filters"],
    install_requires=[
        "numpy>=1.24.2,<1.25",
        "opencv-python>=4.7.0.72,<4.8",
        "perceptual>=0.1,<0.2",
        "scipy>=1.10.1,<1.11"
    ],
    entry_points={},
)
