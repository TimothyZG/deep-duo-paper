import torch
import os
import sys
import pandas as pd
import argparse
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def jointly_calibrate_temperature_trio(logits_l, logits_m, logits_s, labels):
    print("🎯 Joint temperature calibration (Trio) in progress...")
    best_nll = float("inf")
    best_Tl, best_Tm, best_Ts = 1.0, 1.0, 1.0

    for Tl in torch.arange(0.2, 3.2, 0.5):
        for Tm in torch.arange(0.2, 3.2, 0.5):
            for Ts in torch.arange(0.2, 3.2, 0.5):
                logits_avg = (logits_l / Tl + logits_m / Tm + logits_s / Ts) / 3
                nll = F.cross_entropy(logits_avg, labels).item()
                if nll < best_nll:
                    best_nll = nll
                    best_Tl, best_Tm, best_Ts = Tl.item(), Tm.item(), Ts.item()

    print(f"Grid best Tl={best_Tl:.2f}, Tm={best_Tm:.2f}, Ts={best_Ts:.2f}, NLL={best_nll:.4f}")

    Tl = torch.tensor([best_Tl], requires_grad=True, device=logits_l.device)
    Tm = torch.tensor([best_Tm], requires_grad=True, device=logits_m.device)
    Ts = torch.tensor([best_Ts], requires_grad=True, device=logits_s.device)
    optimizer = torch.optim.LBFGS([Tl, Tm, Ts], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        logits_avg = (logits_l / Tl + logits_m / Tm + logits_s / Ts) / 3
        loss = F.cross_entropy(logits_avg, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_Tl, final_Tm, final_Ts = Tl.item(), Tm.item(), Ts.item()
    print(f"Refined Tl={final_Tl:.4f}, Tm={final_Tm:.4f}, Ts={final_Ts:.4f}")
    final_nll = F.cross_entropy((logits_l / Tl + logits_m / Tm + logits_s / Ts)/3, labels).item()
    print(f"Final NLL = {final_nll:.4f}")
    print("✅ Joint calibration complete and models wrapped.")
    return final_Tl, final_Tm, final_Ts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--trio_csv_path", type=str, required=True)
    args = parser.parse_args()

    trio_df = pd.read_csv(args.trio_csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []
    for _, row in trio_df.iterrows():
        print(row['model_large'], row['model_median'], row['model_small'])
        paths = {
            "large": os.path.join(args.prediction_dir_path, f"{row['model_large']}.csv"),
            "median": os.path.join(args.prediction_dir_path, f"{row['model_median']}.csv"),
            "small": os.path.join(args.prediction_dir_path, f"{row['model_small']}.csv")
        }

        if not all(os.path.isfile(p) for p in paths.values()):
            print(f"Skipping trio due to missing files: {paths}")
            continue

        logits_large = torch.tensor(pd.read_csv(paths["large"]).values, dtype=torch.float32, device=device)
        logits_median = torch.tensor(pd.read_csv(paths["median"]).values, dtype=torch.float32, device=device)
        logits_small = torch.tensor(pd.read_csv(paths["small"]).values, dtype=torch.float32, device=device)

        target_path = os.path.join(f"y-prediction/{args.dataset_name}/val", "point_prediction.csv")
        target = torch.tensor(pd.read_csv(target_path)["target"].values, dtype=torch.long, device=device)

        Tl, Tm, Ts = jointly_calibrate_temperature_trio(logits_large, logits_median, logits_small, target)
        all_results.append({
            "model_large": row['model_large'],
            "model_median": row['model_median'],
            "model_small": row['model_small'],
            "temperature_large": Tl,
            "temperature_median": Tm,
            "temperature_small": Ts
        })

    save_dir = f"checkpoints/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "temperature_trio.csv")
    pd.DataFrame(all_results).to_csv(save_path, index=False)
    print(f"✅ Saved calibrated trio temperatures to: {save_path}")

if __name__ == "__main__":
    main()
