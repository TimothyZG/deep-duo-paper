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
    parser.add_argument("--trio_csv_path", required=True)
    parser.add_argument("--prediction_root_dir", required=True)
    parser.add_argument("--save_dir", default="evaluation/eval_res")
    args = parser.parse_args()

    temp_single_df = pd.read_csv(f"checkpoints/{args.dataset_name}/temperature_single_model.csv")
    temp_trio_df = pd.read_csv(f"checkpoints/{args.dataset_name}/temperature_trio.csv")
    trio_info_df = pd.read_csv(args.trio_csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for split in ["test", "ood_test"]:
        split_dir = os.path.join(args.prediction_root_dir, split, "raw")
        if not os.path.isdir(split_dir):
            print(f"⚠️ Split {split} not found, skipping.")
            continue

        label_path = os.path.join(args.prediction_root_dir, split, "point_prediction.csv")
        labels = torch.tensor(pd.read_csv(label_path)["target"].values, dtype=torch.long, device=device)

        all_eval_rows = []

        for idx, row in trio_info_df.iterrows():
            model_l, model_m, model_s = row["model_large"], row["model_median"], row["model_small"]
            print(f"🔁 Trio {idx+1} of {len(trio_info_df)}: {model_l}+{model_m}+{model_s}", flush=True)

            paths = {
                "large": os.path.join(split_dir, f"{model_l}.csv"),
                "median": os.path.join(split_dir, f"{model_m}.csv"),
                "small": os.path.join(split_dir, f"{model_s}.csv"),
            }

            if not all(os.path.exists(p) for p in paths.values()):
                print(f"⚠️ Missing predictions for trio: {model_l}, {model_m}, {model_s}")
                continue

            logits_l = torch.tensor(pd.read_csv(paths["large"]).values, dtype=torch.float32, device=device)
            logits_m = torch.tensor(pd.read_csv(paths["median"]).values, dtype=torch.float32, device=device)
            logits_s = torch.tensor(pd.read_csv(paths["small"]).values, dtype=torch.float32, device=device)
            num_classes = logits_l.shape[1]

            temp_l = temp_single_df[temp_single_df["full_name"] == model_l]["temperature"].values[0]
            temp_m = temp_single_df[temp_single_df["full_name"] == model_m]["temperature"].values[0]
            temp_s = temp_single_df[temp_single_df["full_name"] == model_s]["temperature"].values[0]
            trio_temp_row = temp_trio_df[
                (temp_trio_df["model_large"] == model_l) &
                (temp_trio_df["model_median"] == model_m) &
                (temp_trio_df["model_small"] == model_s)
            ]

            if trio_temp_row.empty:
                print(f"⚠️ Missing joint temperature row for trio: {model_l}, {model_m}, {model_s}")
                continue

            temp_l_joint = trio_temp_row["temperature_large"].values[0]
            temp_m_joint = trio_temp_row["temperature_median"].values[0]
            temp_s_joint = trio_temp_row["temperature_small"].values[0]

            gflops = {
                "large": row["gflops_large"],
                "median": row["gflops_median"],
                "small": row["gflops_small"],
                "gflops_balance": row["gflops_balance"],
                "gflops_balance_sm": row["gflops_balance_sm"]
            }

            logits_avg = (logits_l / temp_l + logits_m / temp_m + logits_s / temp_s) / 3
            probs = F.softmax(logits_avg, dim=1)
            preds = probs.argmax(dim=1)
            uncert_sr = 1 - probs.max(dim=1).values
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)

            for ut, uncert in [("softmax_response", uncert_sr), ("entropy", entropy)]:
                metrics = compute_metrics(probs, preds, labels, uncert, num_classes)
                metrics.update({
                    "model_large": model_l,
                    "model_median": model_m,
                    "model_small": model_s,
                    "mode": "logit_average",
                    "wrapper": "TrioWrapper",
                    "uncertainty_type": ut,
                    "split": split,
                    **gflops
                })
                all_eval_rows.append(metrics)

            logits_joint = (logits_l / temp_l_joint + logits_m / temp_m_joint + logits_s / temp_s_joint) / 3
            probs_joint = F.softmax(logits_joint, dim=1)
            preds_joint = probs_joint.argmax(dim=1)
            uncert_sr_joint = 1 - probs_joint.max(dim=1).values
            entropy_joint = -(probs_joint * (probs_joint + 1e-10).log()).sum(dim=1)

            for ut, uncert in [("softmax_response", uncert_sr_joint), ("entropy", entropy_joint)]:
                metrics = compute_metrics(probs_joint, preds_joint, labels, uncert, num_classes)
                metrics.update({
                    "model_large": model_l,
                    "model_median": model_m,
                    "model_small": model_s,
                    "mode": "temperature_weighted",
                    "wrapper": "TrioWrapper",
                    "uncertainty_type": ut,
                    "split": split,
                    **gflops
                })
                all_eval_rows.append(metrics)
                
            probs_l = F.softmax(logits_l, dim=1)
            preds_l = probs_l.argmax(dim=1)
            for ut, uncert in [("softmax_response", uncert_sr_joint), ("entropy", entropy_joint)]:
                metrics = compute_metrics(probs_l, preds_l, labels, uncert, num_classes)
                metrics.update({
                    "model_large": model_l,
                    "model_median": model_m,
                    "model_small": model_s,
                    "mode": "dictatorial_weighteduncertainty",
                    "wrapper": "TrioWrapper",
                    "uncertainty_type": ut,
                    "split": split,
                    **gflops
                })
                all_eval_rows.append(metrics)

        os.makedirs(args.save_dir, exist_ok=True)
        suffix = "" if split == "test" else "_ood"
        out_path = os.path.join(args.save_dir, f"evaluation_trio_{args.dataset_name}{suffix}.csv")
        pd.DataFrame(all_eval_rows).to_csv(out_path, index=False)
        print(f"✅ Finished evaluation for {split}. Saved to {out_path}")

if __name__ == "__main__":
    main()
