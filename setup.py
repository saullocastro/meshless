import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

VERSION = "0.1.0"

setup(
    name = "meshless",
    version = VERSION,
    author = "Saullo G. P. Castro",
    author_email = "castrosaullo@gmail.com",
    description = ("Meshless Methods for Computational Mechanics"),
    license = "BSD",
    keywords = "es-pim finite element partial diferential equations",
    url = "https://github.com/compmech/meshless",
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=["numpy", "scipy"],
)

with open("./meshless/version.py", "wb") as f:
    f.write(b"__version__ = %s\n" % VERSION.encode())

