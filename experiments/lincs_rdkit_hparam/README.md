This experiment runs in 2 steps:
1. We run a (large) hyperparameter sweep using CCPA with RDKit embeddings. We use the results to pick good hyperparameters for the autoencoder and the adversarial predictors. See `config_lincs_rdkit_hparam_sweep.yaml`.
2. We run a (small) hyperparameter sweep using CCPA with all other embeddings. We sweep just over the embedding-related hparams (drug encoder, drug doser), while fixing the AE & ADV related hparams as selected through (1). See `config_lincs_all_embeddings_hparam_sweep.yaml`.
