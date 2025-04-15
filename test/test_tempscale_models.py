import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image
import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_models.TempScaleWrapper import TempScaleWrapper

def generate_toy_loader(transform, num_classes=10, num_samples=32, batch_size=16):
    torch.manual_seed(42)

    x = []
    for _ in range(num_samples):
        fake_img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        pil_img = to_pil_image(fake_img)
        transformed = transform(pil_img)
        x.append(transformed)

    x = torch.stack(x)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size), num_classes

def collect_logits_and_labels(model, loader, device="cpu"):
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model.predict_with_uncertainty(x)  # 👈 Use this
            uncert = out["uncertainty(softmax_response)"]
            print(f"🔍 Sample uncertainty (1 - softmax response): {uncert[:5]}")
            logits = out["logit"]
            if logits.dim() == 3:  # for shallow ensembles
                logits = logits.mean(dim=1)
            all_logits.append(logits)
            all_labels.append(y)

    return torch.cat(all_logits), torch.cat(all_labels)


def main(csv_path):
    had_error = False  # track any so that tjhe output log has success or failure in its name
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        model_name = row["model_name"]
        source = row["source"]
        tv_weights = row["tv_weights"] if not pd.isna(row["tv_weights"]) else None
        print(f"\n=== Testing {model_name} ({source}) ===")
        num_classes=10
        try:
            # Load model
            model, transform = get_model_with_head(
                model_name=model_name,
                num_classes=num_classes,
                source=source,
                tv_weights=tv_weights,
                freeze=True,
                m_head=1
            )
            loader, _ = generate_toy_loader(transform=transform, num_classes=num_classes)

            # Wrap with TempScaleWrapper
            ts_model = TempScaleWrapper(model)

            # Run forward + calibrate
            logits, labels = collect_logits_and_labels(ts_model, loader)
            ts_model.calibrate_temperature(logits, labels)
            print(f"✅ Temperature calibrated to: {ts_model.temperature.item():.4f}")

        except Exception as e:
            had_error = True
            print(f"❌ Failed to test {model_name}: {e}")
            
    if had_error:
        print("❌ Some models failed.")
        sys.exit(1)
    else:
        print("✅ All models tested successfully.")
        sys.exit(0)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to backbones.csv")
    args = parser.parse_args()
    main(args.csv_path)