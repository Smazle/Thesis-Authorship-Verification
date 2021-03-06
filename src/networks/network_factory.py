#!/usr/bin/python3
# -*- coding: utf-8 -*-

from . import network1
from . import network2
from . import network3
from . import network4
from . import network5
from . import network6
from . import network7
from . import network8
from . import network9
from . import network10
from . import r_network1
from . import r_network2
from . import r_network3
from . import r_network4
from . import r_network5
from . import r_network6
from . import r_network7
from . import r_network8
from . import r_network9
from . import r_network10
from . import r_network11
from enum import Enum


class Network(Enum):
    NETWORK1 = 'network1'
    NETWORK2 = 'network2'
    NETWORK3 = 'network3'
    NETWORK4 = 'network4'
    NETWORK5 = 'network5'
    NETWORK6 = 'network6'
    NETWORK7 = 'network7'
    NETWORK8 = 'network8'
    NETWORK9 = 'network9'
    NETWORK10 = 'network10'
    R_NETWORK1 = 'r_network1'
    R_NETWORK2 = 'r_network2'
    R_NETWORK3 = 'r_network3'
    R_NETWORK4 = 'r_network4'
    R_NETWORK5 = 'r_network5'
    R_NETWORK6 = 'r_network6'
    R_NETWORK7 = 'r_network7'
    R_NETWORK8 = 'r_network8'
    R_NETWORK9 = 'r_network9'
    R_NETWORK10 = 'r_network10'
    R_NETWORK11 = 'r_network11'


def construct_network(network, reader):
    if network == Network.NETWORK1:
        return network1.model(reader)
    elif network == Network.NETWORK2:
        return network2.model(reader)
    elif network == Network.NETWORK3:
        return network3.model(reader)
    elif network == Network.NETWORK4:
        return network4.model(reader)
    elif network == Network.NETWORK5:
        return network5.model(reader)
    elif network == Network.NETWORK6:
        return network6.model(reader)
    elif network == Network.NETWORK7:
        return network7.model(reader)
    elif network == Network.NETWORK8:
        return network8.model(reader)
    elif network == Network.NETWORK9:
        return network9.model(reader)
    elif network == Network.NETWORK10:
        return network10.model(reader)
    elif network == Network.R_NETWORK1:
        return r_network1.model(reader)
    elif network == Network.R_NETWORK2:
        return r_network2.model(reader)
    elif network == Network.R_NETWORK3:
        return r_network3.model(reader)
    elif network == Network.R_NETWORK4:
        return r_network4.model(reader)
    elif network == Network.R_NETWORK5:
        return r_network5.model(reader)
    elif network == Network.R_NETWORK6:
        return r_network6.model(reader)
    elif network == Network.R_NETWORK7:
        return r_network7.model(reader)
    elif network == Network.R_NETWORK8:
        return r_network8.model(reader)
    elif network == Network.R_NETWORK9:
        return r_network9.model(reader)
    elif network == Network.R_NETWORK10:
        return r_network10.model(reader)
    elif network == Network.R_NETWORK11:
        return r_network11.model(reader)
    else:
        raise Exception('Unknown network {}'.format(network))
