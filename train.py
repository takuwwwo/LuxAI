import numpy as np
import json
from pathlib import Path
import os
import sys
import random
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List

from config import *
from agent_constants import *
from model.policy_network import PolicyNetwork
from model.multi_source_net import MultipleAgentsNet
from load_data import create_filepath_dataset_from_json
from dataset import LuxDataset, DatasetOutput


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def take_target_loss(outs: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor,
                     weight: Optional[torch.Tensor] = None):
    b, h, w, out_dim = outs.shape
    outs = outs.reshape(-1, out_dim)

    _, preds = torch.max(outs, dim=1)

    targets = targets.reshape(-1)
    num_targets = torch.sum(targets).item()

    actions = actions.reshape(-1)
    ce_loss_batch = F.cross_entropy(outs, actions, weight=weight, reduce=False) * targets  # 対象外のものは0にする

    loss = torch.sum(ce_loss_batch) / num_targets  # 個別のインスタンスに対して平均を取る
    acc = torch.sum((preds == actions.data) * targets) / num_targets  # ACCの平均を取る
    return loss, acc, num_targets


def collate_fn(batch: List[DatasetOutput]):
    state_arrays = np.array([b.state_array for b in batch])
    action_arrays = np.array([b.action_array for b in batch])
    target_arrays = np.array([b.target_array for b in batch])
    city_action_arrays = np.array([b.city_action_array for b in batch])
    city_target_arrays = np.array([b.city_target_array for b in batch])
    maskings = np.array([b.masking for b in batch])
    agent_labels = np.array([b.agent_label for b in batch])

    return state_arrays, action_arrays, target_arrays, city_action_arrays, city_target_arrays, maskings, agent_labels


def main():
    seed_everything(SEED)

    train_obses_list, train_actions_list, valid_obses_list, valid_actions_list = [], [], [], []
    train_labels_list, valid_labels_list = [], []
    for i, (submission_id, team_name) in enumerate(zip(SUBMISSION_ID_LIST, TEAM_NAME_LIST)):
        episode_dir = f'{DATASET_PATH}/{submission_id}/'
        train_obses, valid_obses = \
            create_filepath_dataset_from_json(episode_dir, team_name=team_name, val_ratio=VAL_RATIO)
        train_obses_list.extend(train_obses)
        valid_obses_list.extend(valid_obses)
        for _ in range(len(train_obses)):
            train_labels_list.append(i)
        for _ in range(len(valid_obses)):
            valid_labels_list.append(i)
        print('train data: ', len(train_obses))
        print('valid data: ', len(valid_obses))

    device = torch.device('cuda:0')
    train_dataset = LuxDataset(train_obses_list, train_labels_list)
    valid_dataset = LuxDataset(valid_obses_list, valid_labels_list)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True)

    torch.backends.cudnn.benchmark = True

    policy_net = PolicyNetwork(in_channels=STATE_CHANNELS, feature_size=FEATURE_SIZE, layers=LAYERS,
                               num_unit_actions=UNIT_ACTIONS, num_citytile_actions=CITYTILE_ACTIONS)
    model = MultipleAgentsNet(policy_net, num_agents=len(SUBMISSION_ID_LIST)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        train_modes = ['train', 'valid'] if VAL_RATIO > 0. else ['train']
        for mode in train_modes:
            if mode == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = valid_dataloader

            epoch_loss, epoch_acc, epoch_targets = 0.0, 0, 0
            epoch_citytile_loss, epoch_citytile_acc, epoch_citytile_targets = 0.0, 0, 0
            weight = torch.Tensor([1., 1., 1., 1., CENTER_WEIGHT, 1., 1., 1., 1., 1.]).to(device)
            citytile_weight = torch.Tensor([1., 1., 1.]).to(device)

            for states, actions, targets, citytile_actions, citytile_targets, maskings, labels in tqdm(dataloader):
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                targets = torch.FloatTensor(targets).to(device)
                citytile_actions = torch.LongTensor(citytile_actions).to(device)
                citytile_targets = torch.FloatTensor(citytile_targets).to(device)
                maskings = torch.BoolTensor(maskings).to(device)
                labels = torch.LongTensor(labels).to(device)

                unit_outs, citytile_outs = model.forward(states, labels, maskings)
                unit_loss, unit_acc, num_unit_targets = take_target_loss(unit_outs, actions, targets, weight)
                citytile_loss, citytile_acc, num_citytile_targets = take_target_loss(citytile_outs, citytile_actions,
                                                                                     citytile_targets, citytile_weight)
                loss = unit_loss + citytile_loss

                epoch_citytile_loss += citytile_loss.item() * num_citytile_targets
                epoch_citytile_acc += citytile_acc * num_citytile_targets
                epoch_citytile_targets += num_citytile_targets

                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_loss += unit_loss.item() * num_unit_targets
                epoch_acc += unit_acc * num_unit_targets
                epoch_targets += num_unit_targets

            epoch_loss = epoch_loss / epoch_targets
            epoch_acc = epoch_acc / epoch_targets
            epoch_citytile_loss = epoch_citytile_loss / epoch_citytile_targets
            epoch_citytile_acc = epoch_citytile_acc / epoch_citytile_targets
            print(
                f'Epoch {epoch + 1}/{NUM_EPOCHS} | {mode} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | '
                f'CLoss: {epoch_citytile_loss:.4f} | CACC: {epoch_citytile_acc:.4f}')

        scheduler.step()
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }, f'checkpoint.pth')

    model_name = f'/home/LuxAI/src/rl/imitation_learning/models/' \
                 f'policy_{SUBMISSION_ID_LIST[0]}_{len(SUBMISSION_ID_LIST)}.pth'
    if torch.cuda.device_count() > 1:
        torch.save(model.module.cpu().policy_net.state_dict(), model_name)
    else:
        torch.save(model.cpu().policy_net.state_dict(), model_name)


if __name__ == '__main__':
    main()