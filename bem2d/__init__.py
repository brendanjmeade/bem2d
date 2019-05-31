import copy
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import fsolve
from scipy.integrate import ode, odeint

matplotlib.rcParams["contour.negative_linestyle"] = "solid"

from .quadratic import *
from .geometry import *
from .plot_fncs import *
from .constant_linear import *
