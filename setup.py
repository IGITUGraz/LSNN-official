"""
The Clear BSD License

Copyright (c) 2019 the LSNN team, institute for theoretical computer science, TU Graz
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of LSNN nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import re

from setuptools import setup, find_packages

__author__ = "Guillaume Bellec, Darjan Salaj, Anand Subramoney, Arjun Rao"
__version__ = "2.0.0"
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
    author_email="guillweb@gmail.com",
    provides=['lsnn'],
    install_requires=requirements,
    dependency_links=dependency_links,
)
