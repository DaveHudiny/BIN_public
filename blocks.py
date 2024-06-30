# Author = David Hudak
# Login = xhudak03
# Subject = BIN
# Year = 2022/2023
# Short Description = cgp_interface.py file of project to subject Biology Inspired Computers. Implements CGP settings and interface.

from numpy import *
import cgp


class DivProtected(cgp.OperatorNode):
    """A node that devides its first by its second input."""
    _arity = 2
    _def_output = "x_0 / (x_1 + 0.000001)"


class Identity(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0"


class AbsSub(cgp.OperatorNode):
    _arity = 2
    _def_output = "abs(x_0 - x_1)"


class Avg(cgp.OperatorNode):
    _arity = 2
    _def_output = "(x_0 + x_1) / 2"


class Sin(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.sin(x_0)"


class Cos(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.cos(x_0)"


class Exp(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.exp(x_0)"


class LogAbsProt(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.log(np.abs(x_0) + 0.00001)"


class Sqrt(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.sqrt(np.abs(x_0))"
