from setuptools import setup, find_packages

setup(
    name="novavista-atlas",
    version="1.0.0",
    description="Intelligent field detection and camera calibration system",
    author="NovaVista",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.21.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
)
