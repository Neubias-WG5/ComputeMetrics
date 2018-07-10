# -*- coding: utf-8 -*-

from setuptools import setup

import neubiaswg5

setup(
    name='Neubias-WG5 Metrics Compute',
    version=neubiaswg5.__version__,
    description='Metric computation module of Neubias-WG5',
    packages=['neubiaswg5', 'neubiaswg5.metrics'],
    install_requires=['tifffile', 'scikit-image', 'scikit-learn']
)
