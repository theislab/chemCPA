# Predicting single-cell perturbation responses for unseen drugs - Notebooks

These notebooks are meant to showcase how to analyse a trained chemCPA model. They also reproduce the results from the paper.

To load the model configs please use the provided `.json` file and define your `load_config` function similar to this:

```python
import json 
from tqdm.auto import tqdm
from chemCPA.paths import PROJECT_DIR

def load_config(seml_collection, model_hash):
    file_path = PROJECT_DIR / f"{seml_collection}.json"  # Provide path to json

    with open(file_path) as f:
        file_data = json.load(f)

    for _config in tqdm(file_data):
        if _config["config_hash"] == model_hash:
            # print(config)
            config = _config["config"]
            config["config_hash"] = _config["config_hash"]
    return config
```

Make sure that the dataset paths are set correctly. Here is how to manually change this in the config:

```python
from chemCPA.paths import DATA_DIR

config["dataset"]["data_params"]["dataset_path"] = DATA_DIR / config["dataset"]["data_params"]["dataset_path"].split('/')[-1]
```

Similarly, the `CHECKPOINT_DIR` should align with the folder where you have stored the trained chemCPA models, this is used in the `utils.py`:

```python
from chemCPA.paths import CHECKPOINT_DIR
```
