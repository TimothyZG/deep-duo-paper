import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_models.TempScaleWrapper import TempScaleWrapper
from load_data.dataloaders import get_dataloaders
from utils.config import load_config
from utils.uncertainty_metrics import compute_ece, compute_risk_coverage_metrics, evaluate_model, compute_metrics
from load_data.datasets import IWildCamDataset, Caltech256Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--source", default="torchvision")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--tv_weights", default="DEFAULT")
    parser.add_argument("--dataset_dir", default="DEFAULT")
    parser.add_argument("--gflops", type=float, required=True)
    parser.add_argument("--model_type", choices=["standard", "uniform_soup", "greedy_soup"], default="standard")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cls = IWildCamDataset if args.dataset_name.lower() == 'iwildcam' else Caltech256Dataset
    num_classes = dataset_cls.num_classes
    
    model_file_name = args.model_name if args.model_type=="standard" else f"{args.model_name}_{args.model_type.split('_')[0]}"
    # Load model and wrap
    ckpt_subdir = "calibrated_ff" if args.model_type == "standard" else "calibrated_soup"
    model_path = f"checkpoints/{args.dataset_name}/{ckpt_subdir}/{model_file_name}.pth"
    
    model, transforms = get_model_with_head(
        model_name=args.model_name,
        num_classes=num_classes,
        source=args.source,
        tv_weights=args.tv_weights,
        freeze=False,
        m_head=1
    )

    dataloaders = get_dataloaders(
        args.dataset_name, args.dataset_dir, 64, 4, transforms
    )
    model = TempScaleWrapper(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    results = evaluate_model(model, dataloaders["test"], device)

    # Compute metrics
    metrics = compute_metrics(results, num_classes)
    metrics.update({
        "model": args.model_name,
        "wrapper": "TempScaleWrapper",
        "source": args.source,
        "uncertainty_type": "softmax_response",
        "gflops": args.gflops,
        "model_type": args.model_type
    })
    eval_path = "evaluation/evaluation_results.csv"
    df = pd.DataFrame([metrics])
    if os.path.exists(eval_path):
        df.to_csv(eval_path, mode='a', index=False, header=False)
    else:
        df.to_csv(eval_path, index=False)
    print(f"✅ Evaluation complete. Metrics appended to {eval_path}")

if __name__ == "__main__":
    main()
