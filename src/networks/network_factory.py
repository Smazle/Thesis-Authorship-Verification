#!/usr/bin/python3

from . import network1
from . import network2
from . import network3
from . import network4
from . import network5
from . import r_network1
from . import r_network2
from . import r_network3
from . import r_network4
from . import r_network5
from enum import Enum

class NetworkFactory:

    def __init__(self):
        pass

    # Load the network we are asked to train.
    def get_network(self, networkname, reader):
        if networkname == 'network1':
            return network1.model(reader)
        elif networkname == 'network2':
            return network2.model(reader)
        elif networkname == 'network3':
            return network3.model(reader)
        elif networkname == 'network4':
            return network4.model(reader)
        elif networkname == 'network5':
            return network4.model(reader)
        elif networkname == 'r_network1':
            return r_network1.model(reader)
        elif networkname == 'r_network2':
            return r_network2.model(reader)
        elif networkname == 'r_network3':
            return r_network3.model(reader)
        elif networkname == 'r_network4':
            return r_network4.model(reader)
        elif networkname == 'r_network5':
            return r_network5.model(reader)
        else:
            raise Exception('Unknown network {}'.format(networkname))


