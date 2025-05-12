import os.path
import os
from .utils.locationencoder import LocationEncoder
import lightning as pl
import optuna
import torch
import pandas as pd
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import MP16Dataset
from utils.G3 import G3
from accelerate import Accelerator, DistributedDataParallelKwargs
from zeroshot_prediction import ZeroShotPredictor
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

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
        hparams_pe["legendre_polys"] = trial.suggest_int("legendre_polys", 10, 30, step=10)
    else:
        raise ValueError(f"Unsupported encoding type: {positional_encoding_type}")

    hparams_nn = {}
    if neural_network_type == "siren":
        hparams_nn["hidden_dim"] = trial.suggest_int("hidden_dim", 256, 1024, step=256)
        hparams_nn["num_layers"] = trial.suggest_int("num_layers", 1, 3)
    elif neural_network_type == "mlp":
        hparams_nn["hidden_dim"] = trial.suggest_int("hidden_dim", 32, 128, step=32)
    elif neural_network_type == "rffmlp":
        hparams_nn["sigma"] = [2**0, 2**4, 2**8]
        hparams_nn["hidden_dim"] = trial.suggest_int("hidden_dim", 32, 128, step=32)
    else:
        raise ValueError(f"Unsupported network type: {neural_network_type}")

    hparams_opt = {}
    hparams_opt["lr"] = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    hparams_opt["wd"] = trial.suggest_float("wd", 5e-7, 5e-6, log=True)

    hparams = {}
    hparams.update(hparams_pe)
    hparams.update(hparams_nn)
    hparams["optimizer"] = hparams_opt
    
    hparams['harmonics_calculation'] = "analytic"
    
    return hparams

def train_1epoch(dataloader, eval_dataloader, earlystopper, model, vision_processor, text_processor, optimizer, scheduler, device, accelerator=None):
    model.train()
    # t = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    # for i, (images, texts, longitude, latitude) in enumerate(t):
    for i, (images, texts, longitude, latitude) in enumerate(dataloader):
        texts = text_processor(text=texts, padding='max_length', truncation=True, return_tensors='pt', max_length=77)
        images = images.to(device)
        texts = texts.to(device)
        longitude = longitude.to(device).float()
        latitude = latitude.to(device).float()
        optimizer.zero_grad()

        output = model(images, texts, longitude, latitude, return_loss=True)
        loss = output['loss']

        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        if i % 1 == 0:
            # t.set_description('step {}, loss {}, lr {}'.format(i, loss.item(), scheduler.get_last_lr()[0]))
            allocated = torch.cuda.memory_allocated(device) / 1024**2  # in MB
            reserved = torch.cuda.memory_reserved(device) / 1024**2    # in MB
            print('step {}/{}, loss {:.4f}, lr {:.6f}, VRAM allocated: {:.2f} MB, reserved: {:.2f} MB'.format(
                i, len(dataloader), loss.item(), scheduler.get_last_lr()[0], allocated, reserved
            ))
    scheduler.step()

def tune(positional_encoding_type, neural_network_type, dataset_name="mp16"):
    n_trials = 10
    timeout = 90 * 60 # seconds
    epochs = 3
    hparams = get_hyperparameter(trial, positional_encoding_type, neural_network_type)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # fine-tune
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = G3(
        device, 
        positional_encoding_type,
        neural_network_type,
        hparams=hparams,
    ).to(device)

    dataset = MP16Dataset(vision_processor = model.vision_processor, text_processor = model.text_processor, image_data_path='/root/.cache/mp-16-images.tar')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=5)


    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            params.append(param)

    optimizer = torch.optim.AdamW([param for name,param in model.named_parameters() if param.requires_grad], lr=3e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.87)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    eval_dataloader = None
    earlystopper = None

    def objective(trial: optuna.trial.Trial) -> float:
        for epoch in range(epochs):
            train_1epoch(dataloader, eval_dataloader, earlystopper, model, model.vision_processor, model.text_processor, optimizer, scheduler, device, accelerator)
            unwrapped_model = accelerator.unwrap_model(model)

            predictor = ZeroShotPredictor(model=unwrapped_model, device=device)
            df, res = predictor.evaluate_im2gps3k(
                df_path="im2gps3k_places365.csv",
                top_k=5
            )
            acc_2500, acc_750, acc_200, acc_25, acc_1 = res

            trial.report(acc_200, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return acc_2500, acc_750, acc_200, acc_25, acc_1

    pruner = optuna.pruners.MedianPruner()
    study_name = f"{dataset_name}-{positional_encoding_type}-{neural_network_type}"
    os.makedirs(f"{TUNE_RESULTS_DIR}/{dataset_name}/runs/", exist_ok=True)
    storage_name = f"sqlite:///{TUNE_RESULTS_DIR}/{dataset_name}/runs/{study_name}.db"
    study = optuna.create_study(study_name=study_name, direction=["maximize", "maximize", "maximize", "maximize", "maximize"], 
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

    runsummary = f"{TUNE_RESULTS_DIR}/{dataset_name}/runs/{positional_encoding_type}-{neural_network_type}.csv"
    os.makedirs(os.path.dirname(runsummary), exist_ok=True)

    study.trials_dataframe().to_csv(runsummary)

def compile_summaries(dataset_name="mp16"):

    tune_results_dir_this_dataset = os.path.join(TUNE_RESULTS_DIR, dataset_name)
    runsdir = os.path.join(tune_results_dir_this_dataset, "runs")

    csvs = [csv for csv in os.listdir(runsdir) if csv.endswith(".csv") and csv != "summary.csv"]

    summary = []

    for csv in csvs:
        df = pd.read_csv(os.path.join(runsdir, csv))
        # Average all 5 objective values to rank (can be changed)
        df["mean_value"] = df[["values_0", "values_1", "values_2", "values_3", "values_4"]].mean(axis=1)
        best_run = df.sort_values(by="mean_value", ascending=False).iloc[0]

        values = {
            "acc@2500": best_run["values_0"],
            "acc@750": best_run["values_1"],
            "acc@200": best_run["values_2"],
            "acc@25": best_run["values_3"],
            "acc@1": best_run["values_4"]
        }

        params = {k.replace("params_", ""): v for k, v in best_run.to_dict().items() if "params_" in k}
        pe, nn = csv.replace(".csv", "").split("-")

        print(f"Best params for {pe}-{nn}:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        row = {"pe": pe, "nn": nn}
        row.update(values)
        row.update(params)
        summary.append(row)

    summary = pd.DataFrame(summary).set_index(["pe", "nn"]).sort_values(by="acc@200", ascending=False)
    summary.to_csv(os.path.join(tune_results_dir_this_dataset, "summary.csv"))

    for acc_type in ["acc@2500", "acc@750", "acc@200", "acc@25", "acc@1"]:
        pivot = pd.pivot_table(summary.reset_index(), index="pe", columns="nn", values=acc_type)
        csv_path = os.path.join(tune_results_dir_this_dataset, f"{acc_type}.csv")
        print(f"Writing {csv_path}")
        pivot.to_csv(csv_path)

        fig, ax = plt.subplots()
        im = ax.imshow(pivot, cmap='viridis')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_xlabel(pivot.columns.name)

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_ylabel(pivot.index.name)
        plt.colorbar(im)

        plt.tight_layout()
        fig.savefig(os.path.join(tune_results_dir_this_dataset, f"{acc_type}.png"),
                    transparent=True, bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
    positional_encoders = ["sh"]
    neural_networks = ["siren"]
    for pe in positional_encoders:
        for nn in neural_networks:
            tune(pe, nn)

    compile_summaries()
