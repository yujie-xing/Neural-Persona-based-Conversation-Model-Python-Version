from params import params
from data import data
from persona import *
from io import open
import string
import numpy as np
import pickle
import linecache

import torch
import torch.nn as nn
from torch.autograd import Variable, backward

parameter = params()
model = persona(parameter)
model.train()