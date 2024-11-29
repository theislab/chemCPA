# This folder contains the hydra config files for the project.

The config files are organized as follows:
main.yaml/lincs.yaml/sciplex.yaml/sciplex_finetune.yaml - the root config files
dataset folder - specifies path to dataset and names of keys into it
training folder - configuration of validation/checkpoint/logging behavior
model folder - configuration of model architecture including the embeddings

For convenience, we provide one main config file for each of the 
main experiments in the paper.


To train on sciplex use the sciplex.yaml root config file.
This config will train the model on sciplex_complete_v2.h5ad which is created in the fifth preprocessing notebook.

To (pre)train on LINCS use the lincs.yaml root config file.
This config will train the model on lincs_full_smiles_sciplex_genes.h5ad as created in the third preprocessing notebook.

To finetune a LINCS model on SciPlex3 use the finetune.yaml root config file.
This utilizes the sciplex_complete_lincs_genes_v2.h5ad dataset created in the fifth preprocessing notebook.
Note, that you need to specify a model hash inside the model folder (config/model/finetune.yaml), pretrained_model_hashed.model field.
This is the name of the folder with the checkpoint inside the training_output folder, which contains the model checkpoints.




