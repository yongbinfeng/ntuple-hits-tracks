import os 
import argparse
from time import time

import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import StepLR

from models.interaction_network import InteractionNetwork
from models.dataset import GraphDataset

device = torch.device("cuda")
model = InteractionNetwork(40).jittable()
path = "/home/ybfeng94/GNN4Tracking/interaction-network/train1_PyG_heptrkx_classic_epoch5_1.5GeV_redo.pt"
model.load_state_dict(torch.load(path))


compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "results.pt")
