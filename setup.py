#!/usr/bin/env python

from setuptools import setup

setup(
   name='mirar',
   version='0.0.0',
   description='Medical Image Reconstruction And Restoration',
   author='Matt Tivnan',
   author_email='tivnanmatt@gmail.com',
   packages=['mirar'],  #same as name
   install_requires=['numpy','scipy','scikit-image'], #external packages as dependencies
)
