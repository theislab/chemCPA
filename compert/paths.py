from pathlib import Path

ROOT = Path(__file__).parent.resolve().parent


DATA_DIR = ROOT / "datasets"
EMBEDDING_DIR = Path(
    "/storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/embeddings"
)
PROJECT_DIR = Path("/storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/")
CHECKPOINT_DIR = Path(
    "/storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/sweeps/checkpoints"
)
FIGURE_DIR = Path("/storage/groups/ml01/projects/2021_chemicalCPA_leon.hetzel/figures/")
