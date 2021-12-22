import torch
from torch import nn

from model.policy_network import PolicyNetwork


class MultipleAgentsNet(torch.nn.Module):
    def __init__(self, policy_net: PolicyNetwork, num_agents: int):
        super().__init__()
        self.policy_net = policy_net
        self.num_agents = num_agents
        self.fc_list = nn.ModuleList(
            [nn.Linear(policy_net.feature_size, policy_net.num_actions) for _ in range(num_agents - 1)])

    def forward(self, states: torch.Tensor, labels: torch.Tensor, maskings: torch.Tensor):
        x = self.policy_net.get_feature(states, maskings)

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # x.shape == (B, H, W, C)
        x = x.reshape(-1, c)  # x.shape == (B * H * W, C)

        out_list = []
        out0 = self.policy_net.fc1(x)  # out0.shape == (B * H * W, out_dim)
        out0 = out0.reshape(b, h, w, -1)  # out0.shape == (B, H, W, out_dim)
        out_list.append(out0.unsqueeze(1))
        for fc in self.fc_list:
            out_i = fc(x)  # out_i.shape == (B * H * W, out_dim)
            out_i = out_i.reshape(b, h, w, -1)  # out_i.shape == (B, H, W, out_dim)
            out_list.append(out_i.unsqueeze(1))
        out = torch.cat(out_list, dim=1)  # out.shape == (B, num_agents, H, W, out_dim)
        labels = labels.reshape(-1, 1, 1, 1, 1)  # labels.shape == (B, 1, 1, 1, 1)
        labels = labels.expand(b, 1, h, w, self.policy_net.num_actions)  # labels.shape == (B, 1, H, W, out_dim)
        out = torch.gather(out, dim=1, index=labels)  # out.shape == (B, 1, H, W, out_dim)
        out = out.squeeze(1)  # out.shape == (B, H, W, out_dim)

        out, out2 = torch.split(out, [self.policy_net.num_unit_actions, self.policy_net.num_citytile_actions], dim=-1)
        return out, out2
