# Predicting single-cell perturbation responses for unseen drugs - Notebooks

These notebooks are meant to showcase how to analyse a trained chemCPA model. They also reproduce the results from the paper. 


As you will not be able to connect to the mongoDB via SEML, you have to use the provided part of the database. To align with the notebooks, simply define your own `load_config` function similar to this: 

```python
import json 
from tqdm.auto import tqdm

def load_config(seml_collection, model_hash):
    file_path = f'{seml_collection}.json' # Provide path to json

    with open(file_path) as f:
        file_data = json.load(f)
    
    for config in tqdm(file_data):
        if config['config_hash']==model_hash:
            config = config['config']
    return config
```

Make sure that the datset paths are set correctly. Here is how to manually change this in the config: 
```python
from chemCPA.paths import DATA_DIR

config["dataset"]["data_params"]["dataset_path"] = DATA_DIR / config["dataset"]["data_params"]["dataset_path"].split('/')[-1]
``` 

Similarly, the `CHECKPOINT_DIR` should align with the folder where you have stored the trained chemCPA models, this is used in the `utils.py`:
```python
from chemCPA.paths import CHECKPOINT_DIR
```
