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

def get_base_model_and_mhead(full_name):
    if "head" in full_name:
        # shallow ensemble
        m_head = int(full_name.split("_")[-1].split("head")[0])
        base_name = "_".join(full_name.split("_")[:-1])
    elif "greedy" in full_name or "uniform" in full_name:
        # soup
        m_head = 1
        base_name = "_".join(full_name.split("_")[:-1])
    else:
        # single model
        m_head = 1
        base_name = full_name
    return base_name, m_head

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--backbone_csv_path", required=True)
    parser.add_argument("--prediction_root_dir", required=True)
    parser.add_argument("--save_dir", default="evaluation/eval_res")
    args = parser.parse_args()

    backbone_df = pd.read_csv(args.backbone_csv_path)
    temp_csv_path = os.path.join("checkpoints", args.dataset_name, "temperature_single_model.csv")
    temp_df = pd.read_csv(temp_csv_path)

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

        for file in os.listdir(split_dir):
            if not file.endswith(".csv"):
                continue
            full_name = file.replace(".csv", "")
            base_name, _ = get_base_model_and_mhead(full_name)
            logits = pd.read_csv(os.path.join(split_dir, file)).values
            logits = torch.tensor(logits, dtype=torch.float32, device=device)

            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            num_classes = probs.shape[1]

            uncert_sr = 1 - probs.max(dim=1).values
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)

            model_info = backbone_df[backbone_df["model_name"] == base_name]
            if model_info.empty:
                print(f"⚠️ No backbone info for {full_name} (base model {base_name}), skipping.")
                continue
            model_info = model_info.iloc[0]

            # Raw evaluation
            for uncertainty_type, uncertainties in [("softmax_response", uncert_sr), ("entropy", entropy)]:
                metrics = compute_metrics(probs, preds, labels, uncertainties, num_classes)
                metrics.update({
                    "model": full_name,
                    "wrapper": "None",
                    "uncertainty_type": uncertainty_type,
                    "gflops": model_info["GFLOPS"],
                    "params": model_info["Params"],
                    "split": split
                })
                all_eval_rows.append(metrics)

            # Temperature scaled evaluation
            temp_row = temp_df[temp_df["full_name"] == full_name]
            if temp_row.empty:
                print(f"⚠️ No temperature found for {full_name}, skipping temp scaling.")
                continue
            T = temp_row["temperature"].values[0]

            logits_temp = logits / T
            probs_temp = F.softmax(logits_temp, dim=1)
            preds_temp = probs_temp.argmax(dim=1)

            uncert_sr_temp = 1 - probs_temp.max(dim=1).values
            entropy_temp = -(probs_temp * (probs_temp + 1e-10).log()).sum(dim=1)

            for uncertainty_type, uncertainties in [("softmax_response", uncert_sr_temp), ("entropy", entropy_temp)]:
                metrics = compute_metrics(probs_temp, preds_temp, labels, uncertainties, num_classes)
                metrics.update({
                    "model": full_name,
                    "wrapper": "TempScaleWrapper",
                    "uncertainty_type": uncertainty_type,
                    "gflops": model_info["GFLOPS"],
                    "params": model_info["Params"],
                    "split": split
                })
                all_eval_rows.append(metrics)

        # Save
        os.makedirs(args.save_dir, exist_ok=True)
        split_suffix = "" if split == "test" else "_ood"
        eval_path = os.path.join(args.save_dir, f"evaluation_single_model_{args.dataset_name}{split_suffix}.csv")
        df = pd.DataFrame(all_eval_rows)
        df.to_csv(eval_path, index=False)
        print(f"✅ Finished evaluation for {split}. Saved to {eval_path}")

if __name__ == "__main__":
    main()
