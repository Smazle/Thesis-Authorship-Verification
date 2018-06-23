#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np


def clean(txt):
    txt = re.sub(r'\$NL\$', '\n', txt)
    txt = re.sub(r'\$SC\$', ';', txt)
    txt = re.sub(r'\$NAME\$', '', txt)
    return txt


def wordProcess(txt):
    txt = re.sub(r'([^\s\w]|_)+', '', txt)
    words = re.split(r'\s', txt)
    words = list(filter(lambda x: x != '', words))

    return words


def add_dim_start(array):
    return np.reshape(array, [1] + list(array.shape))


def add_dim_start_all(arrays):
    return list(map(lambda x: add_dim_start(x), arrays))


def remove_dim_start(array):
    return np.reshape(array, tail(array.shape))


def remove_dim_start_all(arrays):
    return list(map(lambda x: remove_dim_start(x), arrays))


def tail(l):
    return l[1:]


if __name__ == '__main__':
    print(clean('Hello, World!$NL$Hello$NAME$, World!'))
