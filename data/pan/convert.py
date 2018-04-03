#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import sys


output = [['ID', 'Text']]

if len(sys.argv) > 1:
    output[0].insert(1, 'Date')

author = 0
for pan in os.listdir('.'):
    if 'pan' in pan:

        for a in os.listdir('./' + pan + '/'):
            if '.' not in a:

                for sample in os.listdir('./' + pan + '/' + a + '/'):
                    if 'unknown' not in sample:
                        path = './' + pan + '/' + a + '/' + sample
                        txt = open(path, 'r').read()
                        txt = txt.replace('\n', '$NL$')
                        txt = txt.replace(';', '$SC$')
                        txt = txt.replace(u'\ufeff', '')

                        if len(sys.argv) > 1:
                            output.append([str(author),
                                           time.strftime('%d-%m-%y'),
                                           txt])
                        else:
                            output.append([str(author), txt])

                author += 1

output = np.array(output)

if len(sys.argv) > 1:
    np.savetxt('formattedPanWithTime.csv', output,
               delimiter=';', fmt='%s', encoding='utf8')
else:
    np.savetxt('formattedPan.csv', output,
               delimiter=';', fmt='%s', encoding='utf8')
