#!/usr/bin/env python3

import sys
import os

def count(dataPath):
    m = 0
    
    for author in os.listdir(dataPath):
        if "." not in author:
            dataPath_extended = dataPath + author + "/"
            files = [open(dataPath_extended + x, "r").read() for x in os.listdir(dataPath_extended)]
            m = max(m, max(map(lambda x: len(x), files)))
    
    return m

