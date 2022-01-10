'''
    General support functions
'''
import pandas as pd
import numpy as np
import torch
import random
import sys
import os
import matplotlib.pyplot as plt
import logging

from tqdm import tqdm
from copy import deepcopy
from typing import List

def seed_torch(seed=42):
    '''Make the code reproducible'''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True