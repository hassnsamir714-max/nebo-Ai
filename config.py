
import torch

PROJECT_NAME = "Nebo AI"
VERSION = "0.1.0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

VOCAB_SIZE = 8000
MAX_SEQ_LEN = 128

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
D_FF = 1024

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
TRAIN_STEPS = 10000

MODEL_PATH = "checkpoints/nebo_model.pt"
DATA_PATH = "data/train.txt"

print(f"{PROJECT_NAME} v{VERSION}")
print(f"Device: {DEVICE}")
