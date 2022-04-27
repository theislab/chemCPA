
## EXP: `sciplex_hparam`
This experiment is run to determine suitable optimisation hparams for the adversary when fine-tuning on the sciplex dataset. These hparams are meant to be shared when evaluating transfer performace for different drug embedding models. 

Similar to `lincs_rdkit_hparam`, we subset to the `grover_base`, `jtvae`, and `rdkit` embedding to be considerate wrt to compute resources. 

Setup: 
Importantly, we sweep over a split that set some drugs as ood. In this setting the original CPA model is not applicable anymore. The drugs were chose according to the results from the original [sciplex publication](https://www.science.org/doi/full/10.1126/science.aax6234), cf. Fig.S6 in the supplements of the publication. 
