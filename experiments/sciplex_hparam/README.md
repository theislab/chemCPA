
## EXP: `sciplex_hparam`
This experiment is run to determine suitable optimisation hparams for the adversary when fine-tuning on the sciplex dataset. These hparams are meant to be shared when evaluating transfer performace for different drug embedding models. 

Similar to `lincs_rdkit_hparam`, we subset to the `grover_base` and `rdkit` embedding to be considerate wrt to compute resources. 

Setup: 
- Importantly, we sweep over a split that set some drugs as ood. In this setting the vanilla model is not applicable anymore. The drugs were chose according to the results from the original [sciplex publication](https://www.science.org/doi/full/10.1126/science.aax6234) and the ood set includes only drugs that have introduced a significant perturbation, cf. Fig.S6 in the supplements of the publication. 
- Additionally, we include the `split_ho_pathway` split for further validation. Here, only the maximum dosage of some drugs is true ood. Hence, the vanilla model is applicable in this scenario. 
