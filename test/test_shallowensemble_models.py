import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image
import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_models.ShallowEnsembleWrapper import ShallowEnsembleWrapper  # assuming this is defined

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


def collect_uncertainty_metrics(model, loader, device="cpu"):
    model.eval()
    all_preds = []
    all_uncertainties = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model.predict_with_uncertainty(x)
            all_preds.append(out["pred"])
            all_uncertainties.append(out["uncertainty(mutual_information)"])

    preds = torch.cat(all_preds)
    uncertainties = torch.cat(all_uncertainties)
    return preds, uncertainties

def main(csv_path):
    had_error = False
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        model_name = row["model_name"]
        source = row["source"]
        tv_weights = row["tv_weights"] if not pd.isna(row["tv_weights"]) else None
        print(f"\n=== Testing {model_name} ({source}) with Shallow Ensemble ===")
        try:
            num_classes = 10
            m_head = 4  # Shallow ensemble use 4 heads for tests
            model, transform = get_model_with_head(
                model_name=model_name,
                num_classes=num_classes,
                source=source,
                tv_weights=tv_weights,
                freeze=True,
                m_head=m_head
            )
            loader, _ = generate_toy_loader(transform=transform, num_classes=num_classes)

            # Wrap with shallow ensemble
            se_model = ShallowEnsembleWrapper(model)

            # Forward and uncertainty estimation
            preds, uncertainty = collect_uncertainty_metrics(se_model, loader)
            print(f"✅ Predictive mean shape: {preds.shape}, uncertainty shape: {uncertainty.shape}")
            print(f"🔍 Sample uncertainty range: min={uncertainty.min().item():.4f}, max={uncertainty.max().item():.4f}")
        except Exception as e:
            had_error = True
            print(f"❌ Failed to test {model_name}: {e}")

    if had_error:
        print("❌ Some models failed.")
        sys.exit(1)
    else:
        print("✅ All shallow ensemble models tested successfully.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to backbones.csv")
    args = parser.parse_args()
    main(args.csv_path)
