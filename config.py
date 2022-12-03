import json

with open('./SETTINGS.json') as f:
    settings = json.load(f)

PROJECT_PATH = settings['PROJECT_PATH']
DATASET_PATH = settings['DATASET_PATH']
MODEL_DIR = settings['MODEL_DIR']

SEED = 42

# training parameters
NUM_EPOCHS = 100
LAYERS = 18
VAL_RATIO = 0.0
BATCH_SIZE = 64
FEATURE_SIZE = 384
NUM_WORKERS = 14
WEIGHT_DECAY = 1e-05
LR = 1e-03
CENTER_WEIGHT = 0.3

SUBMISSION_ID_LIST = [23297953, 23692494, 23281649, 23825143, 23825329, 23825266, 23825370]
TEAM_NAME_LIST = ['Toad Brigade', 'Toad Brigade', 'Toad Brigade', 'RL is all you need', 'RL is all you need',
                  'RL is all you need', 'RL is all you need']