
import torch
import json
import pickle
import numpy as np
import os
import codecs
import random
from torch import nn
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils import Indexer
import utils
import copy
from constants import *


################
class Model(nn.Module):

    def __init__(self, H, i_size=None, o_size=None, emsize=128, start_idx=1, end_idx=2, typ='encdec', args=None, tie_inp_out_emb=False):
        super(Model, self).__init__()
        pass


    
    

################
class GeneratorModel(nn.Module):

    def __init__(self, H, o_size=None, emsize=128, start_idx=1, end_idx=2, typ='dec', args=None, use_gan=False, unk_idx=None):
        super(GeneratorModel, self).__init__()
        pass
    


####################





class CNN(nn.Module):
    def __init__(self, args, reduced_size=None, info={}):
        super(CNN, self).__init__()
        pass




