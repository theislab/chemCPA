from pprint import pprint

from seml.config import generate_configs, read_config

from compert.seml_sweep_icb import ExperimentWrapper

if __name__ == "__main__":
    exp = ExperimentWrapper(init_all=False)

    # this is how seml loads the config file internally
    seml_config, slurm_config, experiment_config = read_config(
        "simon/config_sciplex3_interactive.yaml"
    )
    # we take the first config generated
    configs = generate_configs(experiment_config)
    assert (
        len(configs) == 1
    ), "Careful, more than one config generated from the yaml file"
    args = configs[0]
    pprint(args)

    exp.seed = 1337
    # loads the dataset splits
    exp.init_dataset(**args["dataset"])

    exp.init_drug_embedding(embedding=args["model"]["embedding"])
    exp.init_model(
        hparams=args["model"]["hparams"],
        additional_params=args["model"]["additional_params"],
        load_pretrained=args["model"]["load_pretrained"],
        append_ae_layer=args["model"]["append_ae_layer"],
        pretrained_model_path=args["model"]["pretrained_model_path"],
        pretrained_model_hashes=args["model"]["pretrained_model_hashes"],
    )
    # setup the torch DataLoader
    exp.update_datasets()

    exp.train(**args["training"])
