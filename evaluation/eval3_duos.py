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
from utils.uncertainty_metrics import compute_metrics, evaluate_model
from load_data.dataloaders import get_dataset_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_large_name", required=True)
    parser.add_argument("--model_small_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--source_small", default="torchvision")
    parser.add_argument("--source_large", default="torchvision")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--ood_dataset_dir", default=None)
    parser.add_argument("--gflops_balance", type=float, required=True)
    parser.add_argument("--gflops_large", type=float, required=True)
    parser.add_argument("--gflops_small", type=float, required=True)
    parser.add_argument("--keep_imagenet_head", action="store_true", help="Keep the original ImageNet head")
    args = parser.parse_args()

    print(f"{args.keep_imagenet_head=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cls = get_dataset_class(args.dataset_name.lower())
    num_classes = dataset_cls.num_classes

    # Load models (TempScaleWrapper first to register temperature param)
    ckpt_root = f"checkpoints/{args.dataset_name}/calibrated_ff"
    if args.dataset_name=="imagenetv2":
        ckpt_root = f"checkpoints/imagenet/calibrated_ff"
    path_large = os.path.join(ckpt_root, f"{args.model_large_name}.pth")
    path_small = os.path.join(ckpt_root, f"{args.model_small_name}.pth")

    model_l, transforms = get_model_with_head(args.model_large_name, num_classes, args.source_large,keep_imagenet_head=args.keep_imagenet_head)
    model_s, _ = get_model_with_head(args.model_small_name, num_classes, args.source_small,keep_imagenet_head=args.keep_imagenet_head)

    model_l = TempScaleWrapper(model_l).to(device)
    model_s = TempScaleWrapper(model_s).to(device)

    model_l.load_state_dict(torch.load(path_large, map_location=device))
    model_s.load_state_dict(torch.load(path_small, map_location=device))

    dataloaders = get_dataloaders(args.dataset_name, args.dataset_dir, batch_size=64, num_workers=4, transforms=transforms, ood_root_dir=args.ood_dataset_dir)

    # results_path = f"evaluation/eval_duo_{args.dataset_name}.csv"
    # all_metrics = []

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

        # results = evaluate_model(duo, dataloaders["test"], device)
        
        for split_name in ["test", "ood_test"]:
            if split_name not in dataloaders or dataloaders[split_name] is None:
                continue
            results = evaluate_model(duo, dataloaders[split_name], device)
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

            eval_suffix = "" if split_name == "test" else "_ood"
            results_path = f"evaluation/evaluation_duo_{args.dataset_name}{eval_suffix}.csv"

            df = pd.DataFrame([metrics])
            if os.path.exists(results_path):
                df.to_csv(results_path, mode='a', index=False, header=False)
            else:
                df.to_csv(results_path, index=False)

            df_all = pd.read_csv(results_path).drop_duplicates()
            df_all.to_csv(results_path, index=False)
            print(f"✅ Evaluation complete. Metrics appended to {results_path}")
    #     metrics = compute_metrics(results, num_classes)
    #     metrics.update({
    #         "mode": mode,
    #         "model_large": args.model_large_name,
    #         "model_small": args.model_small_name,
    #         "wrapper": "DuoWrapper",
    #         "source_large": args.source_large,
    #         "source_small": args.source_small,
    #         "dataset": args.dataset_name,
    #         "gflops_balance": args.gflops_balance,
    #         "gflops_large": args.gflops_large,
    #         "gflops_small": args.gflops_small,
    #     })
    #     all_metrics.append(metrics)


    # df = pd.DataFrame(all_metrics)
    # file_exists = os.path.isfile(results_path)
    # df.to_csv(results_path, mode='a', index=False, header=not file_exists)
    # print(f"✅ All results saved to {results_path}")


if __name__ == "__main__":
    main()