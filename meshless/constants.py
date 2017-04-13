import os
import inspect

inspect.getfile(inspect.currentframe())
abspath = os.path.abspath(inspect.getfile(inspect.currentframe()))
CMHOME = os.path.dirname(abspath)
DOUBLE = 'float64'
