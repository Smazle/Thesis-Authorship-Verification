#!/usr/bin/env
# -*- coding: utf-8 -*-

import re


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


if __name__ == '__main__':
    print(clean('Hello, World!$NL$Hello$NAME$, World!'))
