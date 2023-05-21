"""Setup script for ``polyclonal``."""


import re
import sys

from setuptools import setup


if not (sys.version_info[0] == 3 and sys.version_info[1] >= 8):
    raise RuntimeError(
        "polyclonal requires Python >=3.8.\n"
        f"You are using {sys.version_info[0]}.{sys.version_info[1]}."
    )

# get metadata from package `__init__.py` file as here:
# https://packaging.python.org/guides/single-sourcing-package-version/
metadata = {}
init_file = "polyclonal/__init__.py"
with open(init_file) as f:
    init_text = f.read()
for dataname in ["version", "author", "email", "url"]:
    matches = re.findall("__" + dataname + r'__\s+=\s+[\'"]([^\'"]+)[\'"]', init_text)
    if len(matches) != 1:
        raise ValueError(
            f"found {len(matches)} matches for {dataname} " f"in {init_file}"
        )
    else:
        metadata[dataname] = matches[0]

with open("README.rst") as f:
    readme = f.read()

# main setup command
setup(
    name="polyclonal",
    version=metadata["version"],
    author=metadata["author"],
    author_email=metadata["email"],
    url=metadata["url"],
    download_url="https://github.com/jbloomlab/polyclonal/tarball/"
    + metadata["version"],  # tagged version on GitHub
    description="Model mutational escape from polyclonal antibodies.",
    long_description=readme,
    license="GPLv3",
    install_requires=[
        "altair>=5.0.0",
        "binarymap>=0.6",
        "biopython>=1.79",
        "frozendict>=2.0.7",
        "matplotlib>=3.1",
        "natsort>=8.0",
        "numpy>=1.17",
        "pandas>=1.5",
        "requests",
        "scipy>=1.7.1",
        "urllib3==1.26.15",  # https://github.com/googleapis/python-bigquery/issues/1565
    ],
    platforms="Linux and Mac OS X.",
    packages=["polyclonal"],
    package_dir={"polyclonal": "polyclonal"},
)
