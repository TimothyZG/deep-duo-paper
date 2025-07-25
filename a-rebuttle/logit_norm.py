import pandas as pd
import os
import numpy as np

dataset = "imagenet"
print(f"{dataset=}")

# List all CSV files in the raw logits folder
raw_dir = f"./y-prediction/{dataset}/test/raw"
model_list = [f[:-4] for f in os.listdir(raw_dir) if f.endswith(".csv")]

temperature_df = pd.read_csv(f"./checkpoints/{dataset}/temperature_single_model.csv")
temperature_map = dict(zip(temperature_df["full_name"], temperature_df["temperature"]))
temp_list = [temperature_map[model] for model in model_list]

norm_raw = []
norm_temp_scaled = []

# Compute norm of logits and temperature-scaled logits
for model, temp in zip(model_list, temp_list):
    print(f"Evaluating {model} w/ {temp=}")
    raw_logits_curr = pd.read_csv(f"{raw_dir}/{model}.csv").values
    norm_raw_curr = np.mean(np.linalg.norm(raw_logits_curr, axis=1))
    norm_temp_curr = np.mean(np.linalg.norm(raw_logits_curr / temp, axis=1))
    norm_raw.append(norm_raw_curr)
    norm_temp_scaled.append(norm_temp_curr)

# Save results to a CSV file
df = pd.DataFrame({
    "model_name": model_list,
    "norm_raw": norm_raw,
    "norm_temp_scaled": norm_temp_scaled
})

df.to_csv(f"logit_norm_{dataset}.csv", index=False)
print(f"df saved to logit_norm_{dataset}.csv")


norm_temp_fl_ls = []
norm_temp_fs_ls = []
temperature_duo_df = pd.read_csv(f"./checkpoints/{dataset}/temperature_duo.csv")
for model_large, model_small, temp_fl, temp_fs in temperature_duo_df[["model_large", "model_small", "temperature_large", "temperature_small"]].values:
    print(f"{model_large=} {model_small=} {temp_fl=} {temp_fs=}")
    raw_logits_fl = pd.read_csv(f"{raw_dir}/{model_large}.csv").values
    raw_logits_fs = pd.read_csv(f"{raw_dir}/{model_small}.csv").values
    norm_temp_fl = np.mean(np.linalg.norm(raw_logits_fl / temp_fl, axis=1))
    norm_temp_fs = np.mean(np.linalg.norm(raw_logits_fs / temp_fs, axis=1))
    norm_temp_fl_ls.append(norm_temp_fl)
    norm_temp_fs_ls.append(norm_temp_fs)
    
# Save results to a CSV file
df_duo = pd.DataFrame({
    "fl":temperature_duo_df["model_large"],
    "fs": temperature_duo_df["model_small"],
    "duo_norm_fl": norm_temp_fl_ls,
    "duo_norm_fs": norm_temp_fs_ls
})

df_duo.to_csv(f"duo_logit_norm_{dataset}.csv", index=False)
print(f"df_duo saved to duo_logit_norm_{dataset}.csv")