from matplotlib.pyplot import *
from numpy import *
from numpy.fft import *
from numpy.linalg import *
from numpy.random import *

"""
`pylab` is a historic interface and its use is strongly discouraged. The equivalent
replacement is `matplotlib.pyplot`.  See :ref:`api_interfaces` for a full overview
of Matplotlib interfaces.

`pylab` was designed to support a MATLAB-like way of working with all plotting related
functions directly available in the global namespace. This was achieved through a
wildcard import (``from pylab import *``).

.. warning::
   The use of `pylab` is discouraged for the following reasons:

   ``from pylab import *`` imports all the functions from `matplotlib.pyplot`, `numpy`,
   `numpy.fft`, `numpy.linalg`, and `numpy.random`, and some additional functions into
   the global namespace.

   Such a pattern is considered bad practice in modern python, as it clutters the global
   namespace. Even more severely, in the case of `pylab`, this will overwrite some
   builtin functions (e.g. the builtin `sum` will be replaced by `numpy.sum`), which
   can lead to unexpected behavior.

"""
bytes = ...
abs = ...
bool = ...
max = ...
min = ...
pow = ...
round = ...
