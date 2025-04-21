import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.air import session, Checkpoint
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import wandb
import shutil
import re
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_data.dataloaders import get_dataloaders
from utils.config import load_config
from load_models.ShallowEnsembleWrapper import ShallowEnsembleWrapper

def train_model(config, root_dir, dataset_name, device, model_state=None, finetune=False):
    from load_data.datasets import IWildCamDataset, Caltech256Dataset
    dataset_cls = IWildCamDataset if dataset_name.lower() == 'iwildcam' else Caltech256Dataset
    num_classes = dataset_cls.num_classes

    model, transforms = get_model_with_head(
        model_name=config['model_name'],
        num_classes=num_classes,
        source=config.get('source', 'torchvision'),
        tv_weights=config.get('tv_weights', 'DEFAULT'),
        freeze=not finetune,
        m_head=config.get('m_head', 1)
    )
    if config.get("m_head", 1) > 1:
        model = ShallowEnsembleWrapper(model)
    model.to(device)
    if model_state:
        model.load_state_dict(model_state)

    if finetune:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        os.environ["WANDB_HTTP_TIMEOUT"] = "120"
        wandb.init(project=f"{dataset_name}-{config['model_name']}", config=config, reinit=True)

    dataloaders = get_dataloaders(
        dataset_name, root_dir, config["batch_size"], config["num_workers"], transforms
    )
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    if finetune:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=config["warmup_epochs"]),
                CosineAnnealingLR(optimizer, T_max=config["num_epochs"] - config["warmup_epochs"])
            ],
            milestones=[config["warmup_epochs"]]
        )
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    best_val_metric = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        for batch in dataloaders["train"]:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if isinstance(logits, dict):  # unwrap if model returns a dict
                logits = logits["logit"]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if scheduler: scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloaders["val"]:
                x, y = batch
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        metric = f1_score(all_labels, all_preds, average="macro") if dataset_name == "iwildcam" else accuracy_score(all_labels, all_preds)
        if finetune:
            wandb.log({
                "val_metric": metric,
                "epoch": epoch,
                "train_loss": epoch_loss / len(dataloaders["train"]),
                "lr": scheduler.get_last_lr()[0]
            })

        if metric > best_val_metric:
            best_val_metric = metric
            checkpoint_path = os.path.join("ray_ckpts", session.get_trial_id())
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pth"))
            session.report({"val_metric": metric}, checkpoint=Checkpoint.from_directory(checkpoint_path))
        else:
            session.report({"val_metric": metric})
    if finetune: wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--source", type=str, default="torchvision")
    parser.add_argument("--tv_weights", type=str, default="DEFAULT")
    parser.add_argument("--m_head", type=int, default=1)
    parser.add_argument("--num_samples",type=int, default=12)
    args = parser.parse_args()

    training_cfg = load_config(os.path.join(args.config_dir, f"{args.dataset_name}.yaml"))
    root_dir = args.dataset_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_metric = "val_metric"

    reporter = CLIReporter(
        parameter_columns=['learning_rate', 'weight_decay'],
        metric_columns=[validation_metric, 'training_iteration'],
        max_report_frequency=900
    )
    #################### Linear Probing ####################
    config_lp = {
        "model_name": args.model_name,
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": 0,
        "batch_size": training_cfg["training"]["batch_size"],
        "num_epochs": training_cfg["training"]["num_epochs_lp"],
        "num_workers": training_cfg["training"]["num_workers"],
        "source": args.source,
        "tv_weights": args.tv_weights,
        "m_head": args.m_head,
    }
    result = tune.run(
        tune.with_parameters(
            train_model,
            root_dir=root_dir,
            dataset_name=args.dataset_name,
            device=device,
            finetune=False
        ),
        config=config_lp,
        num_samples=4,
        resources_per_trial={"cpu": 4, "gpu": 1},
        scheduler=ASHAScheduler(max_t=config_lp["num_epochs"],grace_period=config_lp["num_epochs"],metric=validation_metric, mode="max"),
        local_dir="ray_results",
        name="lp_search",
        progress_reporter=reporter,
        log_to_file=True
    )
    best_trial = result.get_best_trial(validation_metric, "max")
    print(f"Best LP metric: {best_trial.last_result[validation_metric]:.4f}")
    best_checkpoint = result.get_best_checkpoint(best_trial, metric=validation_metric, mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        model_path = os.path.join(checkpoint_dir, "model.pth")
        best_state = torch.load(model_path)
        lp_checkpoint_dst = f"checkpoints/{args.dataset_name}/lp/{args.model_name}.pth"
        if args.m_head>1: 
            lp_checkpoint_dst=f"checkpoints/{args.dataset_name}/shallow_ensemble_lp/{args.model_name}_{args.m_head}head.pth"
        os.makedirs(os.path.dirname(lp_checkpoint_dst), exist_ok=True)
        shutil.copy(model_path, lp_checkpoint_dst)
        print(f"✅ Saved best LP model to: {lp_checkpoint_dst}")

    #################### Finetune ####################
    config_ft = {
        "model_name": args.model_name,
        "learning_rate": tune.loguniform(1e-6, 3e-4),
        "weight_decay": tune.loguniform(1e-8, 1e-5),
        "batch_size": training_cfg["training"]["batch_size"],
        "num_epochs": training_cfg["training"]["num_epochs_ff"],
        "num_workers": training_cfg["training"]["num_workers"],
        "grace_period": training_cfg["training"]["grace_period"],
        "warmup_epochs": training_cfg["training"]["warmup_epochs"],
        "source": args.source,
        "tv_weights": args.tv_weights,
        "m_head": args.m_head
    }
    ft_result = tune.run(
        tune.with_parameters(
            train_model,
            root_dir=root_dir,
            dataset_name=args.dataset_name,
            device=device,
            model_state=best_state,
            finetune=True
        ),
        config=config_ft,
        num_samples=args.num_samples,
        resources_per_trial={"cpu": 4, "gpu": 1},
        scheduler=ASHAScheduler(max_t=config_ft["num_epochs"],grace_period=config_ft["grace_period"],metric=validation_metric, mode="max"),
        local_dir="ray_results",
        name="ff_search",
        progress_reporter=reporter,
        raise_on_failed_trial=False,
        fail_fast=False,
        log_to_file=True
    )
    
    # Filter successful trials
    successful_trials = [t for t in ft_result.trials if t.status == "TERMINATED" and t.checkpoint]
    failed_trials = [t for t in ft_result.trials if t not in successful_trials]

    print(f"✅ {len(successful_trials)} successful trials")
    print(f"❌ {len(failed_trials)} failed trials")

    if not successful_trials:
        print("⚠️ No successful trials found. Skipping checkpoint and soup saving.")
        return

    ft_best_trial = max(successful_trials, key=lambda t: t.last_result.get(validation_metric, float("-inf")))
    print(f"🏆 Best FT metric: {ft_best_trial.last_result[validation_metric]:.4f}")

    ft_best_checkpoint = ft_result.get_best_checkpoint(ft_best_trial, metric=validation_metric, mode="max")
    with ft_best_checkpoint.as_directory() as checkpoint_dir:
        model_path = os.path.join(checkpoint_dir, "model.pth")
        best_state = torch.load(model_path)
        ft_checkpoint_dst = f"checkpoints/{args.dataset_name}/ff/{args.model_name}.pth"
        if args.m_head>1: ft_checkpoint_dst=f"checkpoints/{args.dataset_name}/shallow_ensemble_ff/{args.model_name}_{args.m_head}head.pth"
        os.makedirs(os.path.dirname(ft_checkpoint_dst), exist_ok=True)
        shutil.copy(model_path, ft_checkpoint_dst)
        print(f"✅ Saved best FF model to: {ft_checkpoint_dst}")
        
    if args.m_head==1:
        soup_dir = f"checkpoints/{args.dataset_name}/soup/{args.model_name}"
        os.makedirs(soup_dir, exist_ok=True)
        existing = {
            int(re.search(r"(\d+)\.pth$", f).group(1))
            for f in os.listdir(soup_dir) if f.endswith(".pth") and re.search(r"\d+\.pth$", f)
        }
        def get_next_index():
            i = 0
            while i in existing:
                i += 1
            existing.add(i)
            return i

        csv_path = os.path.join(soup_dir, f"{args.model_name}_trials.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model_name", "idx","trial_id", "learning_rate", "weight_decay",
                "batch_size", "val_metric", "checkpoint_path"
            ])
            if write_header:
                writer.writeheader()

            # Save each trial's best checkpoint
            for trial in ft_result.trials:
                if trial.status != "TERMINATED" or not trial.checkpoint:
                    continue
                checkpoint = ft_result.get_best_checkpoint(trial, metric=validation_metric, mode="max")
                if checkpoint:
                    with checkpoint.as_directory() as checkpoint_dir:
                        model_path = os.path.join(checkpoint_dir, "model.pth")
                        idx = get_next_index()
                        soup_path = os.path.join(soup_dir, f"trial{idx}.pth")
                        shutil.copy(model_path, soup_path)
                        print(f"✅ Saved trial checkpoint to: {soup_path}")

                        writer.writerow({
                            "model_name": args.model_name,
                            "idx": idx,
                            "trial_id": trial.trial_id,
                            "learning_rate": trial.config["learning_rate"],
                            "weight_decay": trial.config["weight_decay"],
                            "batch_size": trial.config["batch_size"],
                            "val_metric": trial.last_result[validation_metric],
                            "checkpoint_path": soup_path
                        })

if __name__ == "__main__":
    main()