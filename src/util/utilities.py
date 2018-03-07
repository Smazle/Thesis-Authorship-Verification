#!/usr/bin/env
# -*- coding: utf-8 -*-

import re


def clean(txt):
    txt = re.sub(r'\$NL\$', '\n', txt)
    txt = re.sub(r'\$SC\$', ';', txt)
    return txt
