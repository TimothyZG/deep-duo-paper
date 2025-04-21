import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.TempScaleWrapper import TempScaleWrapper
from load_models.model_loader import get_model_with_head
from load_data.dataloaders import get_dataset_class
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--source", type=str, default="torchvision")
    parser.add_argument("--keep_imagenet_head", action="store_true", help="Keep the original ImageNet head")
    args = parser.parse_args()
    print(f"✔️ keep_imagenet_head = {args.keep_imagenet_head}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    dataset_cls = get_dataset_class(args.dataset_name.lower())
    dataset = dataset_cls(args.dataset_dir)
    num_classes = dataset.num_classes
    
    # 2. Load model & weights
    model, transforms = get_model_with_head(
        model_name=args.model_name,
        num_classes=num_classes,
        source=args.source,
        freeze=False,
        keep_imagenet_head=args.keep_imagenet_head
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model.to(device)

    # 3. Wrap with TempScale
    model = TempScaleWrapper(model)

    # 4. Get DataLoader
    dataloaders = dataset.get_splits(transforms, batch_size=64, num_workers=4)

    # 5. Collect logits + labels
    logits, labels = [], []
    with torch.no_grad():
        for x, y in dataloaders["val"]:
            x, y = x.to(device), y.to(device)
            out = model.model(x)  # unscaled forward
            logits.append(out["logit"] if isinstance(out, dict) else out)
            labels.append(y)
    logits = torch.cat(logits)
    labels = torch.cat(labels)

    # 6. Calibrate
    model.calibrate_temperature(logits, labels)

    # 7. Save wrapped model
    torch.save(model.state_dict(), args.save_path)
    print(f"✅ Temp-scaled model saved to {args.save_path}")

if __name__ == "__main__":
    main()
