import os
import argparse
import torch
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_models.TempScaleWrapper import TempScaleWrapper
from load_models.DuoWrapper import DuoWrapper
from load_data.dataloaders import get_dataloaders
from utils.config import load_config
from utils.uncertainty_metrics import compute_ece, compute_risk_coverage_metrics, compute_metrics, evaluate_model
from load_data.datasets import IWildCamDataset, Caltech256Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_large_name", required=True)
    parser.add_argument("--model_small_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--source_small", default="torchvision")
    parser.add_argument("--source_large", default="torchvision")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--gflops_balance", type=float, required=True)
    parser.add_argument("--gflops_large", type=float, required=True)
    parser.add_argument("--gflops_small", type=float, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cls = IWildCamDataset if args.dataset_name.lower() == 'iwildcam' else Caltech256Dataset
    num_classes = dataset_cls.num_classes

    # Load models (TempScaleWrapper first to register temperature param)
    ckpt_root = f"checkpoints/{args.dataset_name}/calibrated_ff"
    path_large = os.path.join(ckpt_root, f"{args.model_large_name}.pth")
    path_small = os.path.join(ckpt_root, f"{args.model_small_name}.pth")

    model_l, transforms = get_model_with_head(args.model_large_name, num_classes, args.source_large)
    model_s, _ = get_model_with_head(args.model_small_name, num_classes, args.source_small)

    model_l = TempScaleWrapper(model_l).to(device)
    model_s = TempScaleWrapper(model_s).to(device)

    model_l.load_state_dict(torch.load(path_large, map_location=device))
    model_s.load_state_dict(torch.load(path_small, map_location=device))

    dataloaders = get_dataloaders(args.dataset_name, args.dataset_dir, batch_size=64, num_workers=4, transforms=transforms)

    results_path = f"evaluation/eval_duo_{args.dataset_name}.csv"
    all_metrics = []

    for mode in ["softvote", "dictatorial", "confident", "weighted_voting"]:
        print(f"\n🎯 Evaluating Duo mode: {mode}")
        duo = DuoWrapper(model_l, model_s, mode=mode)

        if mode == "weighted_voting":
            print(f"📦 Collecting logits for joint calibration of {args.model_large_name} vs {args.model_small_name}")

            logits_l, logits_s, labels = [], [], []
            model_l.eval()
            model_s.eval()
            with torch.no_grad():
                for x, y in dataloaders["val"]:
                    x, y = x.to(device), y.to(device)
                    logits_l.append(duo.model_large(x))
                    logits_s.append(duo.model_small(x))
                    labels.append(y)

            logits_l = torch.cat(logits_l)
            logits_s = torch.cat(logits_s)
            labels = torch.cat(labels)

            duo.jointly_calibrate_temperature(logits_l, logits_s, labels)

        results = evaluate_model(duo, dataloaders["test"], device)

        metrics = compute_metrics(results, num_classes)
        metrics.update({
            "mode": mode,
            "model_large": args.model_large_name,
            "model_small": args.model_small_name,
            "wrapper": "DuoWrapper",
            "source_large": args.source_large,
            "source_small": args.source_small,
            "dataset": args.dataset_name,
            "gflops_balance": args.gflops_balance,
            "gflops_large": args.gflops_large,
            "gflops_small": args.gflops_small,
        })
        all_metrics.append(metrics)


    df = pd.DataFrame(all_metrics)
    file_exists = os.path.isfile(results_path)
    df.to_csv(results_path, mode='a', index=False, header=not file_exists)
    print(f"✅ All results saved to {results_path}")


if __name__ == "__main__":
    main()