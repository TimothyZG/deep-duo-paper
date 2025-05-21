import torch
import os
import sys
import pandas as pd
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_data.dataloaders import get_dataset_class, get_dataloaders
from load_models.ShallowEnsembleWrapper import ShallowEnsembleWrapper
from load_models.TempScaleWrapper import TempScaleWrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_path", type=str, required=True)
    parser.add_argument("--backbone_csv_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ood_dataset_dir", type=str, default=None)
    parser.add_argument("--keep_imagenet_head", action="store_true", help="Keep the original ImageNet head")
    parser.add_argument("--mode", default="single model", choices=["single model", "soup", "shallow ensemble"],
                    help="Mode for loading models: 'single model', 'soup', or 'shallow ensemble'.")
    args = parser.parse_args()
    print(f"✔️ keep_imagenet_head = {args.keep_imagenet_head}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_df = pd.read_csv(args.backbone_csv_path)
    print("Backbones csv:")
    print(backbone_df.head())
        
    # 1. Load dataset
    dataset_cls = get_dataset_class(args.dataset_name.lower())
    dataset = dataset_cls(args.dataset_dir)
    num_classes = dataset.num_classes
    
    # iterate through available models
    model_paths = [f for f in os.listdir(args.model_dir_path) if os.path.isfile(os.path.join(args.model_dir_path, f))]
    num_head_ls = [1]*len(model_paths)
    if args.mode == "shallow ensemble":
        num_head_ls = [int(f.split("_")[-1].split("head.pth")[0]) for f in model_paths]
    total_model = len(model_paths)
    for i,(model_path,num_head) in enumerate(zip(model_paths,num_head_ls)):
        full_name = model_path.split(".pth")[0]
        # Extract base model_name
        if args.mode in ["shallow ensemble", "soup"]:
            model_name = "_".join(full_name.split("_")[:-1])  # remove last _suffix
        elif args.mode == "single model":
            model_name = full_name
        else:
            raise ValueError(f"Unsupported mode {args.mode}")
        try:
            source = backbone_df[backbone_df["model_name"] == model_name]["source"].values[0]
        except IndexError:
            print(f"backbone df has no {model_name} row")
            continue

        model, transforms = get_model_with_head(
            model_name=model_name,
            num_classes=num_classes,
            source=source,
            freeze=False,
            keep_imagenet_head=args.keep_imagenet_head,
            m_head = num_head
        )
        if args.mode == "shallow ensemble":
            model = ShallowEnsembleWrapper(model)
        ckpt = torch.load(os.path.join(args.model_dir_path, model_path), map_location=device)
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        model.to(device)

        # 3. Get DataLoader
        dataloaders = get_dataloaders(args.dataset_name, 
                                    args.dataset_dir, 
                                    batch_size=32, 
                                    num_workers=4, 
                                    transforms=transforms, 
                                    ood_root_dir=args.ood_dataset_dir)

        for val_set in ["val","test","ood_test"]:
            if val_set not in dataloaders or not dataloaders[val_set]:
                print(f"{args.dataset_name} has no {val_set} set")
                continue
            else:
                logits, labels = [], []
                with torch.no_grad():
                    for x, y in dataloaders[val_set]:
                        x, y = x.to(device), y.to(device)
                        out = model(x)  # unscaled forward
                        logits.append(out["logit"] if isinstance(out, dict) else out)
                        labels.append(y)
                logits = torch.cat(logits)
                labels = torch.cat(labels)

                prediction_save_dir = f"y-prediction/{args.dataset_name}/{val_set}/raw"
                os.makedirs(prediction_save_dir,exist_ok=True)
                prediction_save_path = os.path.join(prediction_save_dir,f"{full_name}.csv")
                pd.DataFrame(logits.cpu().numpy()).to_csv(prediction_save_path,index=False)
                
                point_prediction_save_dir = f"y-prediction/{args.dataset_name}/{val_set}"
                os.makedirs(point_prediction_save_dir,exist_ok=True)
                point_prediction_save_path = os.path.join(point_prediction_save_dir,f"point_prediction.csv")
                point_preds = torch.argmax(logits, dim=1).cpu().numpy()
                if os.path.exists(point_prediction_save_path):
                    df = pd.read_csv(point_prediction_save_path)
                    if "target" not in df.columns:
                        df.insert(0, "target", labels.cpu().numpy())
                    df[full_name] = point_preds
                else:
                    df = pd.DataFrame({
                        "target": labels.cpu().numpy(),
                        full_name: point_preds
                    })
                df.to_csv(point_prediction_save_path, index=False)
        print(f"{i}/{total_model} complete, model name = {model_name}")
                
            
                
if __name__ == "__main__":
    main()
