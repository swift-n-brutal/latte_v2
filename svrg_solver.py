# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:13:18 2017

@author: shiwu_001
"""

import numpy as np
import time

from .config import DTYPE
from .solver import SGDSolver
from .math_func import axpb

class SVRGSolver(SGDSolver):
    def __init__(self, *args, **kwargs):
        super(SVRGSolver, self).__init__(*args, **kwargs)
        self.