from setuptools import setup, find_packages

setup(
    name="ISAS20_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "astropy",
        "tqdm",
    ],
    author="ISAS20 Project Team",
    author_email="your.email@example.com",
    description="Utilities for working with ISAS20 and ARGO data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
) 