import os
import json
import torch
import random
import numpy as np
from datetime import datetime


class BaseConfig:
    _COLAB_PATH = '/content/drive/MyDrive/project'
    _LOCAL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR    = _COLAB_PATH if os.path.exists('/content/drive') else _LOCAL_PATH

    # Data
    DATA_RAW_DIR  = os.path.join(BASE_DIR, 'data/raw')
    DATA_PROC_DIR = os.path.join(BASE_DIR, 'data/processed')
    VOCAB_DIR     = os.path.join(BASE_DIR, 'data/vocab')

    # Outputs
    CKPT_DIR      = os.path.join(BASE_DIR, 'outputs/checkpoints')
    RESULT_DIR    = os.path.join(BASE_DIR, 'outputs/results')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED   = 42


_CONFIGS = {
    'cnn_lstm': {
        'MODEL_TYPE'         : 'lstm',
        'BATCH_SIZE'         : 128,
        'MAX_LEN'            : 50,
        'LEARNING_RATE'      : 3e-4,
        'WEIGHT_DECAY'       : 1e-2,
        'WARMUP_EPOCHS'      : 3,     
        'NUM_EPOCHS'         : 20,    
        'PROJ_WARMUP_EPOCHS' : 0,     
        'EMBED_DIM'          : 512,
        'HIDDEN_DIM'         : 512,
        'DROPOUT'            : 0.3,
        'TEMPERATURE'        : 0.1,
        'HARD_NEG_WEIGHT'    : 0.3,
        'MARGIN'             : 0.3,
        'FREEZE_BACKBONE'    : True,
        'UNFREEZE_FROM'      :'layer3', 
        'UNFREEZE_EPOCH'     : 0,    
        'FINETUNE_LR'        : 3e-5,   
        'CHECKPOINT'         : None,
    },

    'cnn_gru': {
        'MODEL_TYPE'         : 'gru',
        'BATCH_SIZE'         : 128,
        'MAX_LEN'            : 50,
        'LEARNING_RATE'      : 3e-4,
        'WEIGHT_DECAY'       : 1e-2,
        'WARMUP_EPOCHS'      : 3,
        'NUM_EPOCHS'         : 20,
        'PROJ_WARMUP_EPOCHS' : 0,
        'EMBED_DIM'          : 512,
        'HIDDEN_DIM'         : 512,
        'DROPOUT'            : 0.3,
        'TEMPERATURE'        : 0.1,
        'HARD_NEG_WEIGHT'    : 0.3,
        'MARGIN'             : 0.3,
        'FREEZE_BACKBONE'    : True,
        'UNFREEZE_FROM'      :'layer3', 
        'UNFREEZE_EPOCH'     : 0,
        'FINETUNE_LR'        : 3e-5,
        'CHECKPOINT'         : None,
    },

    'cnn_minilm': {
        'MODEL_TYPE'         : 'minilm',
        'BATCH_SIZE'         : 128,
        'MAX_LEN'            : 50,
        'LEARNING_RATE'      : 3e-4,
        'WEIGHT_DECAY'       : 1e-2,
        'WARMUP_EPOCHS'      : 3,
        'NUM_EPOCHS'         : 20,
        'PROJ_WARMUP_EPOCHS' : 0,
        'EMBED_DIM'          : 512,
        'DROPOUT'            : 0.3,
        'TEMPERATURE'        : 0.1,
        'HARD_NEG_WEIGHT'    : 0.3,
        'MARGIN'             : 0.3,
        'FREEZE_BACKBONE'    : True,
        'UNFREEZE_FROM'      :'layer3', 
        'UNFREEZE_EPOCH'     : 0,
        'FINETUNE_LR'        : 3e-5,
        'CHECKPOINT'         : None,
    },
}


_SEARCH_SPACES = {
    'cnn_lstm': {
        'NUM_EPOCHS'      : 10,
        'LEARNING_RATE'   : [3e-4, 5e-4, 1e-3],
        'TEMPERATURE'     : [0.07, 0.1],
        'HARD_NEG_WEIGHT' : [0.2, 0.3],
    },
    'cnn_gru': {
        'NUM_EPOCHS'      : 10,
        'LEARNING_RATE'   : [3e-4, 5e-4, 1e-3],
        'TEMPERATURE'     : [0.07, 0.1],
        'HARD_NEG_WEIGHT' : [0.2, 0.3],
    },
    'cnn_minilm': {
        'NUM_EPOCHS'      : 10,
        'LEARNING_RATE'   : [3e-4, 5e-4, 1e-3],
        'TEMPERATURE'     : [0.07, 0.1],
        'HARD_NEG_WEIGHT' : [0.2, 0.3],
    },
}


# -------------------- Checkpoint Registry --------------------
def get_save_path(model_name, val_r1=None, cfg_tag=''):
    """Tạo path mới + ghi vào registry"""
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag  = f'_{cfg_tag}' if cfg_tag else ''
    name = f'{model_name}{tag}_{ts}.pth'
    path = os.path.join(BaseConfig.CKPT_DIR, name)
    _register_checkpoint(model_name, path, val_r1)
    print(f"--> [Checkpoint] Save path: {path}")
    return path


def _register_checkpoint(model_name, path, val_r1=None):
    registry_path = os.path.join(BaseConfig.CKPT_DIR, 'registry.json')

    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {}

    if model_name not in registry:
        registry[model_name] = []

    registry[model_name].append({
        'path'   : path,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'val_r1' : val_r1,
    })

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


def update_checkpoint_r1(model_name, path, val_r1):
    """Cập nhật R@1 cho checkpoint sau khi train xong"""
    registry_path = os.path.join(BaseConfig.CKPT_DIR, 'registry.json')
    if not os.path.exists(registry_path):
        return

    with open(registry_path) as f:
        registry = json.load(f)

    for ckpt in registry.get(model_name, []):
        if ckpt['path'] == path:
            ckpt['val_r1'] = val_r1
            break

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"--> [Registry] Updated R@1={val_r1}% for {os.path.basename(path)}")


def get_latest_checkpoint(model_name):
    """Lấy checkpoint mới nhất"""
    registry_path = os.path.join(BaseConfig.CKPT_DIR, 'registry.json')
    if not os.path.exists(registry_path):
        print("Chưa có checkpoint nào.")
        return None

    with open(registry_path) as f:
        registry = json.load(f)

    if model_name not in registry or not registry[model_name]:
        print(f"Chưa có checkpoint cho {model_name}.")
        return None

    latest = registry[model_name][-1]
    print(f"--> [Checkpoint] Latest {model_name}: {latest['path']} "
          f"| R@1: {latest['val_r1']}% | {latest['created']}")
    return latest['path']


def get_best_checkpoint(model_name):
    """Lấy checkpoint có R@1 cao nhất"""
    registry_path = os.path.join(BaseConfig.CKPT_DIR, 'registry.json')
    if not os.path.exists(registry_path):
        print("Chưa có checkpoint nào.")
        return None

    with open(registry_path) as f:
        registry = json.load(f)

    if model_name not in registry or not registry[model_name]:
        print(f"Chưa có checkpoint cho {model_name}.")
        return None

    valid = [c for c in registry[model_name] if c['val_r1'] is not None]
    if not valid:
        return get_latest_checkpoint(model_name)

    best = max(valid, key=lambda x: x['val_r1'])
    print(f"--> [Checkpoint] Best {model_name}: {best['path']} "
          f"| R@1: {best['val_r1']}% | {best['created']}")
    return best['path']


def list_checkpoints(model_name=None):
    """In toàn bộ checkpoint"""
    registry_path = os.path.join(BaseConfig.CKPT_DIR, 'registry.json')
    if not os.path.exists(registry_path):
        print("Chưa có checkpoint nào.")
        return

    with open(registry_path) as f:
        registry = json.load(f)

    models = [model_name] if model_name else registry.keys()
    for name in models:
        if name not in registry:
            continue
        print(f"\n{name}:")
        for i, ckpt in enumerate(registry[name]):
            marker = ' ← latest' if i == len(registry[name]) - 1 else ''
            r1     = f"R@1: {ckpt['val_r1']}%" if ckpt['val_r1'] else 'R@1: N/A'
            print(f"  [{i+1}] {r1} | {ckpt['created']} | {ckpt['path']}{marker}")


# -------------------- Helpers --------------------
def get_config(name):
    if name not in _CONFIGS:
        raise ValueError(f"Config '{name}' không tồn tại! Chọn: {list(_CONFIGS.keys())}")
    return _CONFIGS[name].copy()


def get_search_space(name):
    if name not in _SEARCH_SPACES:
        raise ValueError(f"Search space '{name}' không tồn tại!")
    return _SEARCH_SPACES[name]


def set_global_seed(seed=BaseConfig.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


set_global_seed()