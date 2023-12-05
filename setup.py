#!/usr/bin/env python

from os import path
from setuptools import find_packages, setup


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "improv", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    return version

PROJECTS = {}

setup(
    name="improv",
    version=get_version(),
    author="Jiarui Xu",
    url="https://github.com/xvjiarui/IMProv",
    description="Diffuser for Video generation",
    packages=find_packages(exclude=("configs", "tests*")) + list(PROJECTS.keys()),
    package_dir=PROJECTS,
    python_requires=">=3.6",
    install_requires=[
        "einops>=0.3.2",
        "diffusers[torch]>=0.12.1",
        "transformers==4.26.1",
    ],
)
