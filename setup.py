# setup.py
from setuptools import setup, find_packages

setup(
    name="trading-ml",              # whatever you want to call it
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
