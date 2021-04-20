# This file is part of SliceRL by M. Rossi

from __future__ import print_function
import sys
from setuptools import setup, find_packages

if sys.version_info < (3,6):
    print("SliceRL requires Python 3.6 or later", file=sys.stderr)
    sys.exit(1)

with open('README.md') as f:
    long_desc = f.read()

setup(name= "slicerl",
      version = '1.0.0',
      description = "Slicing Problem with Reinforcement Learning for DUNE reconstruction",
      author = "M. Rossi",
      author_email = "marco.rossi@cern.ch",
      url="https://github.com/marcorossi5/SlicingRL.git",
      long_description = long_desc,
      entry_points = {'console_scripts':
                      ['slicerl = slicerl.scripts.slicerl:main']},
      package_dir = {'': 'src'},
      packages = find_packages('src'),
      zip_safe = False,
      classifiers=[
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            ],
     )
