"""
This file installs the LSNN package.
Copyright (C) 2019 the LSNN team, TU Graz
"""
import re

from setuptools import setup, find_packages

__author__ = "Guillaume Bellec, Darjan Salaj, Anand Subramoney, Arjun Rao"
__version__ = "1.0.8dev"
__description__ = """
    Tensorflow code for Recurrent Spiking Neural Network (so called Long short term memory Spiking Neural Network, LSNN).
    
    For more details:
    Long short-term memory and learning-to-learn in networks of spiking neurons
    Guillaume Bellec*, Darjan Salaj*, Anand Subramoney*, Robert Legenstein, Wolfgang Maass
    * equal contributions
    
    Authors contributions:
    GB implemented the initial code of the spiking RNN in Tensorflow. DS lead most of the sequential-MNIST simulation.
    AS added features to the model and helped making the code distributable. AR helped to optimize the code.
"""
__copyright__ = "Copyright (C) 2019 the LSNN team"
__license__ = "The Clear BSD License"


def get_requirements(filename):
    """
    Helper function to read the list of requirements from a file
    """
    dependency_links = []
    with open(filename) as requirements_file:
        requirements = requirements_file.read().strip('\n').splitlines()
    requirements = [req for req in requirements if not req.startswith('#')]
    for i, req in enumerate(requirements):

        if ':' in req:
            match_obj = re.match(r"git\+(?:https|ssh|http):.*#egg=(.*)-(.*)", req)
            assert match_obj, "Cannot make sense of url {}".format(req)
            requirements[i] = "{req}=={ver}".format(req=match_obj.group(1), ver=match_obj.group(2))
            dependency_links.append(req)
    return requirements, dependency_links

requirements, dependency_links = get_requirements('requirements.txt')

setup(
    name="LSNN",
    version=__version__,
    packages=find_packages('.'),
    author=__author__,
    description=__description__,
    license=__copyright__,
    copyright=__copyright__,
    author_email="bellec@igi.tugraz.at",
    provides=['lsnn'],
    install_requires=requirements,
    dependency_links=dependency_links,
)
