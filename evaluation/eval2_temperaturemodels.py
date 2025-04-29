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
from utils.uncertainty_metrics import evaluate_model, compute_metrics
from load_data.dataloaders import get_dataset_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--source", default="torchvision")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--tv_weights", default="DEFAULT")
    parser.add_argument("--dataset_dir", default="DEFAULT")
    parser.add_argument("--gflops", type=float, required=True)
    parser.add_argument("--model_type", choices=["standard", "uniform_soup", "greedy_soup"], default="standard")
    parser.add_argument("--keep_imagenet_head", action="store_true", help="Keep the original ImageNet head")
    args = parser.parse_args()
    print(f"✔️ keep_imagenet_head = {args.keep_imagenet_head}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cls = get_dataset_class(args.dataset_name.lower())
    num_classes = dataset_cls.num_classes
    
    model_file_name = args.model_name if args.model_type=="standard" else f"{args.model_name}_{args.model_type.split('_')[0]}"
    # Load model and wrap
    ckpt_subdir = "calibrated_ff" if args.model_type == "standard" else "calibrated_soup"
    model_path = f"checkpoints/{args.dataset_name}/{ckpt_subdir}/{model_file_name}.pth"
    if args.dataset_name=="imagenetv2":
        model_path = f"checkpoints/imagenet/{ckpt_subdir}/{model_file_name}.pth"
    
    model, transforms = get_model_with_head(
        model_name=args.model_name,
        num_classes=num_classes,
        source=args.source,
        tv_weights=args.tv_weights,
        freeze=False,
        m_head=1,
        keep_imagenet_head=args.keep_imagenet_head
    )

    dataloaders = get_dataloaders(
        args.dataset_name, args.dataset_dir, 64, 4, transforms
    )
    model = TempScaleWrapper(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    for split_name in ["test", "ood_test"]:
        if split_name not in dataloaders or dataloaders[split_name] is None:
            continue
        results = evaluate_model(model, dataloaders[split_name], device)
        label_path = f"label_check_{args.dataset_name}.csv"
        if not os.path.exists(label_path):
            label_df = pd.DataFrame({"label": results["labels"].numpy()})
            label_df.to_csv(label_path, index=False)
            print(f"✅ Saved labels to {label_path}")
        else:
            print(f"ℹ️ Label file {label_path} already exists, skipping save.")
        metrics = compute_metrics(results, num_classes)
        metrics.update({
            "model": args.model_name,
            "wrapper": "TempScaleWrapper",
            "source": args.source,
            "uncertainty_type": "softmax_response",
            "gflops": args.gflops,
            "model_type": args.model_type
        })

        eval_suffix = "" if split_name == "test" else "_ood"
        eval_path = f"evaluation/evaluation_single_model_{args.dataset_name}{eval_suffix}.csv"

        df = pd.DataFrame([metrics])
        if os.path.exists(eval_path):
            df.to_csv(eval_path, mode='a', index=False, header=False)
        else:
            df.to_csv(eval_path, index=False)

        df_all = pd.read_csv(eval_path).drop_duplicates()
        df_all.to_csv(eval_path, index=False)
        print(f"✅ Evaluation complete. Metrics appended to {eval_path}")

if __name__ == "__main__":
    main()
