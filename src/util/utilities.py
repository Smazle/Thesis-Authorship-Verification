#!/usr/bin/env
# -*- coding: utf-8 -*-

import re


def clean(txt):
    txt = re.sub(r'\$NL\$', '\n', txt)
    txt = re.sub(r'\$SC\$', ';', txt)
    return txt


def wordProcess(txt):
    txt = txt.replace('\n', ' ')
    txt = txt.split(' ')
    txt = [''.join(list(filter(lambda x: x.isalnum(), q))) for q in txt]
    txt = list(filter(lambda x: x != '', txt))
    return txt
