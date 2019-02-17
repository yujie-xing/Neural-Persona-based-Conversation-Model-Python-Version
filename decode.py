from decode_params import decode_params
from data import data
from persona import *
from decode_model import decode_model
from io import open
import string
import numpy as np
import pickle
import linecache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, backward


parameter = decode_params()
model = decode_model(parameter)
model.decode()
