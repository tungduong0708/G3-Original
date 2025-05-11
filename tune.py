import os.path
from .utils.locationencoder import LocationEncoder
import lightning as pl
import optuna
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import pandas as pd
import yaml

TUNE_RESULTS_DIR = "results/tune"

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

def get_hyperparameter(trial: optuna.trial.Trial, positional_encoding_type, neural_network_type):

    hparams_pe = {}
    if positional_encoding_type == "projectionrff":
        hparams_pe["projection"] = trial.suggest_categorical("projection", ["ecef", "mercator", "eep"])
        hparams_pe["sigma"] = [2**0, 2**4, 2**8]
    elif positional_encoding_type == "projection":
        hparams_pe["projection"] = trial.suggest_categorical("projection", ["ecef", "mercator", "eep"])
    elif positional_encoding_type == "sh":
        hparams_pe["legendre_polys"] = trial.suggest_int("legendre_polys", 10, 30, step=5)
        hparams_pe["embedding_dim"] = trial.suggest_int("embedding_dim", 16, 128, step=16)
    else:
        raise ValueError(f"Unsupported encoding type: {positional_encoding_type}")

    hparams_nn = {}
    if neural_network_type == "siren":
        hparams_nn["hidden_dim"] = trial.suggest_int("hidden_dim", 32, 128, step=32)
        hparams_nn["num_layers"] = trial.suggest_int("num_layers", 1, 3)
    elif neural_network_type == "mlp":
        hparams_nn["hidden_dim"] = trial.suggest_int("hidden_dim", 32, 128, step=32)
    elif neural_network_type == "rffmlp":
        hparams_nn["sigma"] = [2**0, 2**4, 2**8]
        hparams_nn["hidden_dim"] = trial.suggest_int("hidden_dim", 32, 128, step=32)
    else:
        raise ValueError(f"Unsupported network type: {neural_network_type}")

    hparams_opt = {}
    hparams_opt["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hparams_opt["wd"] = trial.suggest_float("wd", 1e-8, 1e-1, log=True)

    hparams = {}
    hparams.update(hparams_pe)
    hparams.update(hparams_nn)
    hparams["optimizer"] = hparams_opt
    
    hparams['harmonics_calculation'] = "analytic"
    
    return hparams

def tune(positional_encoding_type, neural_network_type):
    n_trials = 100
    timeout = 4 * 60 * 60 # seconds
    epochs = 100

    if dataset == "landoceandataset":
        datamodule = LandOceanDataModule()
        num_classes = 1
        regression = False
        presence_only = False
        loss_bg_weight = False
    elif dataset == "checkerboard":
        datamodule = CheckerboardDataModule()
        num_classes = 16
        regression = False
        presence_only = False
        loss_bg_weight = False,
    elif dataset == "inat2018":
        datamodule = Inat2018DataModule("/data/sphericalharmonics/inat2018/")
        num_classes = 8142
        regression = False
        presence_only = True
        loss_bg_weight = 5
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    def objective(trial: optuna.trial.Trial) -> float:

        hparams = get_hyperparameter(trial, positional_encoding_type, neural_network_type)
        hparams["num_classes"] = num_classes
        hparams["presence_only_loss"] = presence_only
        hparams["loss_bg_weight"] = loss_bg_weight
        hparams["regression"] = regression

        spatialencoder = LocationEncoder(
                            positional_encoding_type,
                            neural_network_type,
                            hparams=hparams
            )

        trainer = pl.Trainer(
            max_epochs=epochs,
            log_every_n_steps=5,
            accelerator='gpu', 
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)])

        trainer.logger.log_hyperparams(hparams)

        trainer.fit(model=spatialencoder, datamodule=datamodule)

        return trainer.callback_metrics["val_loss"].item()

    pruner = optuna.pruners.MedianPruner()
    study_name = f"{dataset}-{positional_encoding_type}-{neural_network_type}"
    os.makedirs(f"{TUNE_RESULTS_DIR}/{dataset}/runs/", exist_ok=True)
    storage_name = f"sqlite:///{TUNE_RESULTS_DIR}/{dataset}/runs/{study_name}.db"
    study = optuna.create_study(study_name=study_name, direction="minimize", 
                                storage=storage_name, load_if_exists=True, 
                                pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    study.trials_dataframe()

    runsummary = f"{TUNE_RESULTS_DIR}/{dataset}/runs/{positional_encoding_type}-{neural_network_type}.csv"
    os.makedirs(os.path.dirname(runsummary), exist_ok=True)

    study.trials_dataframe().to_csv(runsummary)

def compile_summaries(dataset):
    tune_results_dir_this_datset = os.path.join(TUNE_RESULTS_DIR, dataset)
    runsdir = os.path.join(TUNE_RESULTS_DIR, f"{dataset}/runs")

    csvs = [csv for csv in os.listdir(runsdir) if csv.endswith("csv") and csv != "summary.csv"]

    summary = []
    hparams = {}
    for csv in csvs:
        df = pd.read_csv(os.path.join(runsdir, csv))
        best_run = df.sort_values(by="value").iloc[0]
        value = best_run.value
        params = {k.replace("params_", ""): v for k, v in best_run.to_dict().items() if "params" in k}
        pe, nn = csv.replace(".csv", "").split("-")
        hparams[f"{pe}-{nn}"] = params

        sum = {
            "pe":pe,
            "nn":nn,
            "value":value
        }
        sum.update(params)

        summary.append(sum)

    summary = pd.DataFrame(summary).sort_values("value").set_index(["pe","nn"])
    summary.to_csv(os.path.join(tune_results_dir_this_datset, "summary.csv"))

    print("writing " + os.path.join(tune_results_dir_this_datset, "hparams.yaml"))
    with open(os.path.join(tune_results_dir_this_datset, "hparams.yaml"), 'w') as f:
        yaml.dump(hparams, f)

    value_matrix = pd.pivot_table(summary.value.reset_index(), index="pe", columns="nn", values=["value"])["value"]
    print("writing " + os.path.join(tune_results_dir_this_datset, "values.csv"))
    value_matrix.to_csv(os.path.join(tune_results_dir_this_datset, "values.csv"))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(value_matrix)
    ax.set_xticks(range(len(value_matrix.columns)))
    ax.set_xticklabels(value_matrix.columns)
    ax.set_xlabel(value_matrix.columns.name)

    ax.set_yticks(range(len(value_matrix.index)))
    ax.set_yticklabels(value_matrix.index)
    ax.set_ylabel(value_matrix.index.name)

    plt.tight_layout()

    print("writing "+os.path.join(tune_results_dir_this_datset, "values.png"))
    fig.savefig(os.path.join(tune_results_dir_this_datset, "values.png"), transparent=True, bbox_inches="tight", pad_inches=0)

if __name__ == '__main__':
    #positional_encoders = ["theory", "direct", "cartesian3d", "grid"] # "sphericalharmonics",
    #neural_networks = ["siren", "fcnet", "linear", "mlp"]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="landoceandataset", help="Name of the dataset")

    args = parser.parse_args()

    dataset = args.dataset

    positional_encoders = ["cartesian3d", "sphericalharmonics"]
    neural_networks = ["linear", "siren", "fcnet"]
    for pe in positional_encoders:
        for nn in neural_networks:
            tune(pe, nn, dataset=dataset)

    compile_summaries(dataset)
