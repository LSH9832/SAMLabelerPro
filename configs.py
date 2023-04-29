import yaml
from enum import Enum
import os

DEFAULT_TITLE = "SAMLabeler Pro"

DEFAULT_CONFIG_FILE = 'settings/init.yaml'
CONFIG_FILE = 'settings/default.yaml'
REMOTE_CONFIG_FILE = 'settings/remote.yaml'
DEFAULT_EDIT_CONFIG = "cache/last_edit.yaml"

os.makedirs("cache", exist_ok=True)
os.makedirs("settings", exist_ok=True)

def load_config(file):
    with open(file, 'rb')as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg

def save_config(cfg, file):
    s = yaml.dump(cfg)
    with open(file, 'w') as f:
        f.write(s)
    return True

class STATUSMode(Enum):
    VIEW = 0
    CREATE = 1
    EDIT = 2

class DRAWMode(Enum):
    POLYGON = 0
    SEGMENTANYTHING = 1

class CLICKMode(Enum):
    POSITIVE = 0
    NEGATIVE = 1

class MAPMode(Enum):
    LABEL = 0
    SEMANTIC = 1
    INSTANCE = 2
