# Generalized Wrapped Gaussian Mixture Model
# author: https://github.com/ludlows
# 2021-Oct
# file: setup.py
"""
An implementation of the (Generalized) Wrapped Gaussian Mixture Model in the thesis below.

 Reference:
        @mastersthesis{wang2020speech,
        title={Speech Enhancement using Fiber Acoustic Sensor},
        author={Wang, Miao},
        year={2020},
        school={Concordia University},
        url={https://spectrum.library.concordia.ca/id/eprint/986722/1/Miao_MASc_S2020.pdf}
        }
"""

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="wgmm",
    version="0.0.0",
    author="Miao Wang",
    description="Wrapped Gaussian Mixture Model (WGMM) for angular clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ludlows/wgmm",
    setup_requires=['setuptools', 'numpy'],
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ]
)
