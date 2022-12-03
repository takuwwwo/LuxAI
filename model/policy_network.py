from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from agent_constants import MAP_SIZE


class FReLU(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                              groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor):
        return torch.max(x, self.bn(self.conv(x)))


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.sigmoid(self.fc2(torch.relu_(self.fc1(y))))
        y = y.reshape(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicConv2d(nn.Module):
    """
    This class refers to https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
    """
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, bn: bool, use_se: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None
        self.use_se = use_se
        if use_se:
            self.se_layer = SELayer(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        if self.use_se:
            h = self.se_layer(h)

        return h


class LuxStateNet(nn.Module):
    """
    This class refers to https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
    """
    def __init__(self, in_channels: int, feature_size: int, layers: int):
        super().__init__()
        self.conv0 = BasicConv2d(in_channels, feature_size, (3, 3), True, use_se=False)
        self.blocks = nn.ModuleList([BasicConv2d(feature_size, feature_size, (3, 3), True, use_se=True) for _ in range(layers)])
        self.frelu_list = nn.ModuleList([FReLU(feature_size) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu_(self.conv0(x))
        for block, frelu in zip(self.blocks, self.frelu_list):
            h = frelu(h + block(h))
        return h


class StateNetwork(nn.Module):
    def __init__(self, in_channels: int, feature_size: int, layers: int, map_size: int = MAP_SIZE):
        super().__init__()
        self.net = LuxStateNet(in_channels, layers=layers, feature_size=feature_size)
        self.feature_size = feature_size
        self.transformer_feature_size = feature_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1,
                                                        dim_feedforward=self.transformer_feature_size)
        self.pos_embedding = nn.Parameter(torch.randn(map_size * map_size, 1, self.transformer_feature_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, masking: torch.Tensor) -> torch.Tensor:
        # masking.shape == (B, H, W)

        x = self.net(x)  # x.shape == (B, C, H, W)
        b, c, h, w = x.shape
        masking = masking.reshape(b, h * w)

        x = x.permute(0, 2, 3, 1)  # x.shape == (B, H, W, C)
        y = x.reshape(b, h * w, c).permute(1, 0, 2)  # y.shape == (H * W, B, C)
        y = self.dropout(y + self.pos_embedding)  # y.shape == (H * W, B, C)
        y = self.encoder_layer(y, src_key_padding_mask=masking)  # y.shape == (H * W, B, C)
        y = y.permute(1, 0, 2).reshape(b, h, w, c)  # y.shape == (B, H, W, C)

        x = x + y  # x.shape == (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # x.shape == (B, C, H, W)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, in_channels: int, feature_size: int, layers: int, num_unit_actions: int,
                 num_citytile_actions: int):
        super().__init__()
        self.state_net = StateNetwork(in_channels, feature_size, layers)

        self.feature_size = feature_size
        self.num_actions = num_unit_actions + num_citytile_actions
        self.num_unit_actions = num_unit_actions
        self.num_citytile_actions = num_citytile_actions

        self.fc1 = nn.Linear(self.feature_size, self.num_actions)

    def forward(self, states: torch.Tensor, maskings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.get_feature(states, maskings)

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # x.shape == (B, H, W, C)
        x = x.reshape(-1, c)  # x.shape == (B * H * W, C)

        out = self.fc1(x)  # out.shape == (B * H * W, out_dim)
        out = out.reshape(b, h, w, -1)  # out.shape == (B, H, W, out_dim)

        out, out2 = torch.split(out, [self.num_unit_actions, self.num_citytile_actions], dim=-1)
        return out, out2

    def get_feature(self, states: torch.Tensor, maskings: torch.Tensor) -> torch.Tensor:
        return self.state_net(states, maskings)

    def act(self, states: torch.Tensor, maskings: torch.Tensor, targets: torch.Tensor):
        def _get_action_index_and_probs(x: torch.Tensor):
            probs = F.softmax(x, dim=-1)  # x.shape == (B, H, W, out_dim)
            actions = x.argmax(dim=-1)  # actions.shape == (B, H, W)

            log_prob = self.get_log_prob(probs, actions, targets)
            return actions, probs, log_prob
        # targets.shape == (B, H, W)
        x, y = self.forward(states, maskings)
        return _get_action_index_and_probs(x), _get_action_index_and_probs(y)

    def get_log_prob(self, probs: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor):
        probs = torch.gather(probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # probs.shape == (B, H, W)

        referenced = torch.ones_like(probs)
        probs = torch.where(targets == 1, probs, referenced)

        prob = torch.prod(probs)
        log_prob = torch.log(prob)

        return log_prob