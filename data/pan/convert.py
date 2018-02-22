#!/usr/bin/env python3

import numpy as np
import os

output = [['ID', 'Text']]

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

                        output.append([str(author), txt])

            author += 1

output = np.array(output)
np.savetxt('formattedPan.csv', output,
           delimiter=';', fmt='%s', encoding='utf8')
