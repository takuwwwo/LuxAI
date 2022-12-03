from typing import Dict
import sys
import os

# For Only Submission

path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'

import torch

from agent_ import Agent, STATE_CHANNELS
from model.policy_network import PolicyNetwork
from config import MODEL_DIR

device = torch.device('cpu')
policy_net = PolicyNetwork(in_channels=STATE_CHANNELS, feature_size=384, layers=18, num_unit_actions=10,
                           num_citytile_actions=3)
policy_net.load_state_dict(torch.load(f'{MODEL_DIR}/model.pth'))
policy_net.eval()
agent_ = Agent(policy_net, device, research_th=0.05, research_turn=15)


def agent(observation, configuration):
    return agent_(observation, configuration)

