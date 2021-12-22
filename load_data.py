import json
from pathlib import Path
import random


def create_filepath_dataset_from_json(episode_dir: str, val_ratio: float, team_name: str):
    obses = dict()

    episodes = [path for path in Path(episode_dir).glob('**/*.json') if 'info' not in path.name]
    for filepath in episodes:
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        if json_load['info']['TeamNames'][0] == team_name:
            index = 0
        elif json_load['info']['TeamNames'][1] == team_name:
            index = 1
        else:
            continue

        obses[ep_id] = dict()
        for i in range(len(json_load['steps']) - 1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                obses[ep_id][i] = (filepath, i, index)

    # split
    episode_list = list(obses.keys())
    random.shuffle(episode_list)
    num_train = int(len(episode_list) * (1. - val_ratio))
    train_list, valid_list = episode_list[:num_train], episode_list[num_train:]
    train_obses, valid_obses = [], []
    for episode in train_list:
        for filepath, turn, player_id in obses[episode].values():
            train_obses.append((filepath, turn, player_id))
    for episode in valid_list:
        for filepath, turn, player_id in obses[episode].values():
            valid_obses.append((filepath, turn, player_id))

    return train_obses, valid_obses