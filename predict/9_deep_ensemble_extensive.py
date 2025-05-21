import os
import sys
import argparse
import itertools
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
    brier = brier_score_loss(y_true=np.eye(num_classes)[labels].reshape(-1), y_pred=probs.reshape(-1))
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
    parser.add_argument("--backbone_csv_path", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--save_dir", default="evaluation/eval_res")
    parser.add_argument("--max_member", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    label_path = os.path.normpath(os.path.join(args.prediction_dir, "..", "point_prediction.csv"))
    labels = pd.read_csv(label_path)["target"]
    labels = torch.tensor(labels.values, dtype=torch.long, device=device)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"evaluation_ensemble_combinations_{args.dataset_name}.csv")

    def append_to_csv(row_dicts):
        df_out = pd.DataFrame(row_dicts)
        file_exists = os.path.exists(save_path)
        file_has_data = os.path.getsize(save_path) > 0 if file_exists else False
        df_out.to_csv(save_path, mode="a", index=False, header=not file_has_data)

    # Load all logit files
    all_logits = {}
    for file in sorted(os.listdir(args.prediction_dir)):
        if not file.endswith(".csv"):
            continue
        model_name = file.replace(".csv", "")
        logits = pd.read_csv(os.path.join(args.prediction_dir, file), header=None).values
        all_logits[model_name] = torch.tensor(logits, dtype=torch.float32, device=device)

    num_classes = list(all_logits.values())[0].shape[1]

    # Evaluate ensembles from size 1 to max_member
    model_keys = list(all_logits.keys())
    for m in range(1, args.max_member + 1):
        if len(model_keys) < m:
            continue
        for members in itertools.combinations(model_keys, m):
            member_logits = [all_logits[m] for m in members]
            logits_avg = torch.stack(member_logits).mean(dim=0)

            probs = F.softmax(logits_avg, dim=1)
            preds = probs.argmax(dim=1)
            uncert_sr = 1 - probs.max(dim=1).values
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)

            row_dicts = []
            for uncertainty_type, uncertainties in [("softmax_response", uncert_sr), ("entropy", entropy)]:
                metrics = compute_metrics(probs, preds, labels, uncertainties, num_classes)
                metrics.update({
                    "ensemble_size": m,
                    "wrapper": "LogitAverage",
                    "uncertainty_type": uncertainty_type,
                    "members": ",".join(members),
                    "split": "test"
                })
                row_dicts.append(metrics)

            append_to_csv(row_dicts)
            print(f"✅ Saved evaluation for ensemble of size {m} with members: {members}")

if __name__ == "__main__":
    main()
