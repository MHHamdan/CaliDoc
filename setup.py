"""
CaliDoc: Calibrated Document Understanding with Uncertainty Quantification

Setup script for installation and distribution.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="calidoc",
    version="0.1.0",
    author="Mohammed Hamdan",
    author_email="",
    description="Calibrated Document Understanding with Uncertainty Quantification and Robustness Guarantees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MHHamdan/CaliDoc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "calidoc-train=scripts.train:main",
            "calidoc-eval=scripts.evaluate:main",
        ],
    },
)
