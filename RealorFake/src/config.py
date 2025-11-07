import torch
import random
import numpy as np
import os

# --- Environment and Paths ---
# We assume data.zip is in the root, and we extract to ./data/
ZIP_FILE_PATH = "./data.zip"
EXTRACT_DIR = "./data"

# Paths inside the extracted folder
KAGGLE_TRAIN_DIR = os.path.join(EXTRACT_DIR, "data/train")
KAGGLE_TEST_DIR = os.path.join(EXTRACT_DIR, "data/test")
KAGGLE_TRAIN_CSV = os.path.join(EXTRACT_DIR, "data/train.csv")

# --- Global Settings ---
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Cross-Encoder (RoBERTa) Model Config ---
MODEL_NAME = "roberta-base"
MAX_LEN = 512
PER_SIDE = 250   # tokens for each side; remaining for specials
CE_EPOCHS = 3
CE_LR = 2e-5
CE_BATCH_SIZE = 8
CE_FOLDS = 5

# --- Blender (LightGBM) Config ---
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=20,
    max_depth=-1,
    verbosity=-1,
    seed=SEED
)
LGB_ROUNDS = 2000
LGB_FOLDS = 5

# --- Utility Function ---
def set_seed(seed=SEED):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_device_info():
    """Prints information about the execution device."""
    print(f"--- Device Information ---")
    if torch.cuda.is_available():
        print(f"GPU available: True")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"GPU available: False")
        print(f"Device: CPU only")
    print(f"Using device: {DEVICE}")