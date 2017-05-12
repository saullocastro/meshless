import os
import inspect

import numpy as np

_abspath = os.path.abspath(inspect.getfile(inspect.currentframe()))
MESHLESSHOME = os.path.dirname(_abspath)
DOUBLE = 'float64'
XGLOBAL = np.array([1., 0, 0])
YGLOBAL = np.array([0, 1., 0])
ZGLOBAL = np.array([0, 0, 1.])
