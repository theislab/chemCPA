from pathlib import Path

ROOT = Path(__file__).parent.resolve().parent

PROJECT_DIR = ROOT
DATA_DIR = PROJECT_DIR / "data"
EMBEDDING_DIR = PROJECT_DIR
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
FIGURE_DIR = PROJECT_DIR / "figures"
WB_DIR = PROJECT_DIR / "wandb"
