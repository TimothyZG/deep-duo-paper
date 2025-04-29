import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss, log_loss, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.uncertainty_metrics import compute_ece, compute_risk_coverage_metrics

def compute_metrics(probs, preds, labels, uncertainties, num_classes):
    probs = probs.cpu().numpy()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    uncertainties = uncertainties.cpu().numpy()

    certainties = 1 - uncertainties
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    brier = brier_score_loss(
        y_true=np.eye(num_classes)[labels].reshape(-1),
        y_proba=probs.reshape(-1)
    )
    nll = log_loss(labels, probs, labels=list(range(num_classes)))
    ece = compute_ece(torch.tensor(probs), torch.tensor(labels))

    cp_auroc = roc_auc_score((preds == labels).astype(int), certainties)
    aurc, eaurc, sac_dict = compute_risk_coverage_metrics(labels, preds, uncertainties)

    metrics = {
        "Acc": acc,
        "F1": f1,
        "Brier": brier,
        "NLL": nll,
        "ECE": ece,
        "CP_AUROC": cp_auroc,
        "AURC": aurc,
        "E-AURC": eaurc,
    }
    metrics.update({f"SAC@{int(k*100)}": v for k, v in sac_dict.items()})
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--duo_csv_path", required=True)
    parser.add_argument("--prediction_root_dir", required=True)
    parser.add_argument("--save_dir", default="evaluation")
    args = parser.parse_args()

    temp_single_df = pd.read_csv(f"checkpoints/{args.dataset_name}/temperature_single_model.csv")
    temp_duo_df = pd.read_csv(f"checkpoints/{args.dataset_name}/temperature_duo.csv")
    duo_info_df = pd.read_csv(args.duo_csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for split in ["test", "ood_test"]:
        split_dir = os.path.join(args.prediction_root_dir, split, "raw")
        if not os.path.isdir(split_dir):
            print(f"⚠️ Split {split} not found, skipping.")
            continue

        label_path = os.path.join(args.prediction_root_dir, split, "point_prediction.csv")
        labels = pd.read_csv(label_path)["target"]
        labels = torch.tensor(labels.values, dtype=torch.long, device=device)

        all_eval_rows = []

        for idx, row in duo_info_df.iterrows():
            model_large = row["model_large"]
            model_small = row["model_small"]

            path_large = os.path.join(split_dir, f"{model_large}.csv")
            path_small = os.path.join(split_dir, f"{model_small}.csv")

            if not (os.path.exists(path_large) and os.path.exists(path_small)):
                print(f"⚠️ Missing prediction files for {model_large} or {model_small}, skipping.")
                continue

            logits_large = torch.tensor(pd.read_csv(path_large).values, dtype=torch.float32, device=device)
            logits_small = torch.tensor(pd.read_csv(path_small).values, dtype=torch.float32, device=device)

            num_classes = logits_large.shape[1]

            # Fetch temperatures
            temp_row_l = temp_single_df[temp_single_df["full_name"] == model_large]
            temp_row_s = temp_single_df[temp_single_df["full_name"] == model_small]
            temp_duo_row = temp_duo_df[
                (temp_duo_df["model_large"] == model_large) &
                (temp_duo_df["model_small"] == model_small)
            ]

            if temp_row_l.empty or temp_row_s.empty or temp_duo_row.empty:
                print(f"⚠️ Missing temperature info for {model_large} and {model_small}, skipping.")
                continue

            temp_l = temp_row_l["temperature"].values[0]
            temp_s = temp_row_s["temperature"].values[0]
            temp_duo_l = temp_duo_row["temperature_large"].values[0]
            temp_duo_s = temp_duo_row["temperature_small"].values[0]

            # Fetch gflops info
            gflops_large = row["gflops_large"]
            gflops_small = row["gflops_small"]
            gflops_balance = row["gflops_balance"]

            #### 1. Logit Average Duo (with single model temp scaling first)
            logits_avg = (logits_large / temp_l + logits_small / temp_s) / 2
            probs = F.softmax(logits_avg, dim=1)
            preds = probs.argmax(dim=1)
            uncert_sr = 1 - probs.max(dim=1).values
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)

            for uncertainty_type, uncertainties in [("softmax_response", uncert_sr), ("entropy", entropy)]:
                metrics = compute_metrics(probs, preds, labels, uncertainties, num_classes)
                metrics.update({
                    "model_large": model_large,
                    "model_small": model_small,
                    "mode": "logit_average",
                    "wrapper": "DuoWrapper",
                    "uncertainty_type": uncertainty_type,
                    "gflops_large": gflops_large,
                    "gflops_small": gflops_small,
                    "gflops_balance": gflops_balance,
                    "split": split
                })
                all_eval_rows.append(metrics)

            #### 2. Dictatorial Duo (large model's prediction, use different uncertainty)
            preds_dictatorial = logits_large.argmax(dim=1)

            # Use uncertainties from:
            #   (a) logit average
            for uncertainty_type, uncertainties in [("softmax_response", uncert_sr), ("entropy", entropy)]:
                metrics = compute_metrics(probs, preds_dictatorial, labels, uncertainties, num_classes)
                metrics.update({
                    "model_large": model_large,
                    "model_small": model_small,
                    "mode": "dictatorial_avguncertainty",
                    "wrapper": "DuoWrapper",
                    "uncertainty_type": uncertainty_type,
                    "gflops_large": gflops_large,
                    "gflops_small": gflops_small,
                    "gflops_balance": gflops_balance,
                    "split": split
                })
                all_eval_rows.append(metrics)

            #   (b) temperature-weighted logits
            logits_weighted = (logits_large / temp_duo_l + logits_small / temp_duo_s) / 2
            probs_weighted = F.softmax(logits_weighted, dim=1)
            uncert_sr_weighted = 1 - probs_weighted.max(dim=1).values
            entropy_weighted = -(probs_weighted * (probs_weighted + 1e-10).log()).sum(dim=1)

            for uncertainty_type, uncertainties in [("softmax_response", uncert_sr_weighted), ("entropy", entropy_weighted)]:
                metrics = compute_metrics(probs_weighted, preds_dictatorial, labels, uncertainties, num_classes)
                metrics.update({
                    "model_large": model_large,
                    "model_small": model_small,
                    "mode": "dictatorial_weighteduncertainty",
                    "wrapper": "DuoWrapper",
                    "uncertainty_type": uncertainty_type,
                    "gflops_large": gflops_large,
                    "gflops_small": gflops_small,
                    "gflops_balance": gflops_balance,
                    "split": split
                })
                all_eval_rows.append(metrics)

            #### 3. Temperature Weighted Duo (joint calibration)
            preds_weighted = probs_weighted.argmax(dim=1)
            for uncertainty_type, uncertainties in [("softmax_response", uncert_sr_weighted), ("entropy", entropy_weighted)]:
                metrics = compute_metrics(probs_weighted, preds_weighted, labels, uncertainties, num_classes)
                metrics.update({
                    "model_large": model_large,
                    "model_small": model_small,
                    "mode": "temperature_weighted",
                    "wrapper": "DuoWrapper",
                    "uncertainty_type": uncertainty_type,
                    "gflops_large": gflops_large,
                    "gflops_small": gflops_small,
                    "gflops_balance": gflops_balance,
                    "split": split
                })
                all_eval_rows.append(metrics)

        # Save
        os.makedirs(args.save_dir, exist_ok=True)
        split_suffix = "" if split == "test" else "_ood"
        eval_path = os.path.join(args.save_dir, f"evaluation_duo_{args.dataset_name}{split_suffix}.csv")
        df = pd.DataFrame(all_eval_rows)
        df.to_csv(eval_path, index=False)
        print(f"✅ Finished evaluation for {split}. Saved to {eval_path}")

if __name__ == "__main__":
    main()
