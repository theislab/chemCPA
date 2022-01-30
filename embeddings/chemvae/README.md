# Training the Chemical VAE (w/ Teacher forcing)

The files need to contain a single SMILES per row, no trailing comma.
Header needs to be `SMILES`.

We run with the standard hyperparameters, except for the max KL weight (β), which we set to `1.0` to get
a more disentangled latent space.

Dimension of embedding: 128

```bash
# header
echo "SMILES" > ../lincs_trapnell_zinc.csv
# add train SMILES to file
tail -n +2 ../lincs_trapnell.smiles >> ../lincs_trapnell_zinc.csv # skip the first line (`smiles`)
cat ../zinc_smiles_train.csv >> ../lincs_trapnell_zincs.csv
# start the training
./train_chemvae.sh
```