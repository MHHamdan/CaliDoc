"""
CaliDoc: Calibrated Field-Level Risk and Annotation Auditing for Document
Information Extraction (WACV 2027).

`pip install .` installs what inference with a released checkpoint needs;
`pip install ".[research]"` adds the training and analysis stack in
requirements.txt.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    research = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="calidoc",
    version="0.1.0",
    author="Mohammed Hamdan",
    description="Calibrated field-level risk and annotation auditing for document information extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MHHamdan/CaliDoc",
    packages=find_packages(include=["calidoc", "calidoc.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "huggingface-hub>=0.19.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "research": research,
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "calidoc-predict=calidoc.predict:main",
        ],
    },
)
