from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="leaf-net",
    version="0.1.0",
    author="Ayush Shukla",
    author_email="shuklaayush0704@gmail.com",
    description="A lightweight, Transformer-like architecture built on 1D-Convolutions for energy-efficient computer vision.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kingknight07/Leaf-net",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "huggingface_hub>=0.20.0"
    ],
)
