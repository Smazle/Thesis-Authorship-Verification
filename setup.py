#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from setuptools import setup, find_packages
import os

requirements = open('./requirements', 'r').read()
requirements = requirements.split('[APT]\n')
requirements = [[y for y in x.split('\n') if y != ''] for x in requirements]

pip = requirements[0][1:]
bash = requirements[1]

for i in bash:
    os.system('sudo apt-get install ' + i)

for i in pip:
    os.system('sudo pip3 install ' + i)

# setup(name='MastersThesis',
#      version='0.1',
#      url='https://github.com/smazle/MastersThesis',
#      author='Smazle',
#      install_requires=pip,
#      packages=["src.networks"],
#      scripts=["src/networks/network1.py"]
#      )
# os.system("sudo rm -rf *.egg-info")
