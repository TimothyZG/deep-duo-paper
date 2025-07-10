# experiment.py

import os
import argparse
import sys
import torch
from TargetMatching import TargetMatching
from multi_transform_dataloader import MultiTransformDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, log_loss, roc_auc_score


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_data.dataloaders import get_dataloaders,IWildCamDataset, Caltech256Dataset
from load_models.DuoWrapper import DuoWrapper
from utils.uncertainty_metrics import compute_ece, compute_risk_coverage_metrics

def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    original_nll = F.cross_entropy(logits, labels).item()
    print(f"NLL before temperature scaling = {original_nll:.4f}")
    best_nll = float("inf")
    best_T = 1.0

    for T in torch.arange(0.05, 5.05, 0.05):
        T = T.item()
        loss = F.cross_entropy(logits / T, labels).item()
        if loss < best_nll:
            best_nll = loss
            best_T = T
    print(f"Grid search best T = {best_T:.3f}, NLL = {best_nll:.4f}")

    print(f"Use LBFGS to find a fine-grained temperature")
    temp_tensor = torch.tensor([best_T], requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([temp_tensor], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temp_tensor, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_refined = temp_tensor.detach().item()

    final_nll = F.cross_entropy(logits / T_refined, labels).item()
    print(f"Refined T = {T_refined:.4f}")
    print(f"NLL after temperature scaling = {final_nll:.4f}")
    return T_refined


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

def evaluate_with_temp_scaling(student, val_loader, test_loader, device):
    print("📏 Running temperature calibration on validation set...")
    student.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                x = batch["student_inputs"]
                y = batch["labels"]
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            logits = student(x)
            logits_list.append(logits)
            labels_list.append(y)

    logits_val = torch.cat(logits_list, dim=0)
    labels_val = torch.cat(labels_list, dim=0)
    # Calibrate
    T = calibrate_temperature(logits_val, labels_val)
    print(f"🔧 Optimal temperature: {T:.3f}")

    # --- Evaluation on Test Set ---
    print("🔍 Evaluating on test set with and without temperature scaling...")
    test_logits, test_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                x = batch["student_inputs"]
                y = batch["labels"]
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            logits = student(x)
            test_logits.append(logits)
            test_labels.append(y)

    logits_test = torch.cat(test_logits, dim=0)
    labels_test = torch.cat(test_labels, dim=0)
    probs = F.softmax(logits_test, dim=1)
    preds = probs.argmax(dim=1)
    num_classes = probs.shape[1]

    # Uncertainty measures
    uncert_sr = 1 - probs.max(dim=1).values
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)

    print("✅ Raw Model Evaluation:")
    raw_metrics = compute_metrics(probs, preds, labels_test, uncert_sr, num_classes)
    for k, v in raw_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Temp-scaled
    probs_temp = F.softmax(logits_test / T, dim=1)
    preds_temp = probs_temp.argmax(dim=1)
    uncert_sr_temp = 1 - probs_temp.max(dim=1).values
    entropy_temp = -(probs_temp * (probs_temp + 1e-10).log()).sum(dim=1)

    print("✅ Temp-Scaled Evaluation:")
    temp_metrics = compute_metrics(probs_temp, preds_temp, labels_test, uncert_sr_temp, num_classes)
    for k, v in temp_metrics.items():
        print(f"  {k}: {v:.4f}")

    return {"raw": raw_metrics, "temp_scaled": temp_metrics, "temperature": T}
    

def create_duo_teacher_with_calibration(teacher_fl_name, teacher_fl_path, teacher_fl_source,
                                       teacher_fs_name, teacher_fs_path, teacher_fs_source,
                                       num_classes, device, val_loader, mode="asymmetric"):
    """
    Create a DuoWrapper teacher with joint temperature calibration
    """
    # Load both teacher models
    teacher_fl, transforms_fl = get_model_with_head(
        model_name=teacher_fl_name,
        num_classes=num_classes,
        source=teacher_fl_source,
    )
    
    teacher_fs, transforms_fs = get_model_with_head(
        model_name=teacher_fs_name,
        num_classes=num_classes,
        source=teacher_fs_source,
    )
    
    # Move to device and load weights
    teacher_fl.to(device)
    teacher_fs.to(device)
    teacher_fl.load_state_dict(torch.load(teacher_fl_path, map_location=device))
    teacher_fs.load_state_dict(torch.load(teacher_fs_path, map_location=device))
    
    # Create DuoWrapper
    duo_teacher = DuoWrapper(teacher_fl, teacher_fs, mode=mode)
    duo_teacher.to(device)
    
    # Find joint temperatures if using asymmetric mode
    if mode == "asymmetric":
        duo_teacher.find_joint_temperatures(val_loader, device)
    
    # Return teacher transforms (could be FL transforms or a combination)
    teacher_transforms = (transforms_fl,transforms_fs)
    
    return duo_teacher, teacher_transforms

def create_multi_transform_dataloader(dataset_name, dataset_dir, transforms_dict, 
                                    batch_size=32, num_workers=4, split='train'):
    """
    Create a dataloader that applies different transforms to the same data
    """
    # Get the base dataset without transforms
    if dataset_name.lower() == 'iwildcam':
        dataset_cls = IWildCamDataset
    else:
        dataset_cls = Caltech256Dataset
    
    # You'll need to modify this based on how your get_dataloaders works
    # This is a placeholder - you might need to access the raw dataset directly
    base_dataloaders = get_dataloaders(dataset_name, dataset_dir,
                                     batch_size=batch_size, num_workers=num_workers,
                                     transforms=None)  # No transforms initially
    base_dataset = base_dataloaders[split].dataset
    
    # Create multi-transform dataset
    multi_dataset = MultiTransformDataset(base_dataset, transforms_dict)
    
    # Create dataloader
    dataloader = DataLoader(multi_dataset, batch_size=batch_size, 
                          num_workers=num_workers, shuffle=(split=='train'))
    
    return dataloader

# Updated main function section
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--teacher_fl_name", type=str, required=True)
    parser.add_argument("--teacher_fl_path", type=str, required=True)
    parser.add_argument("--teacher_fl_source", type=str, default="torchvision")
    parser.add_argument("--teacher_fs_name", type=str, required=True)
    parser.add_argument("--teacher_fs_path", type=str, required=True)
    parser.add_argument("--teacher_fs_source", type=str, default="torchvision")
    parser.add_argument("--student_name", type=str, required=True)
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--student_source", type=str, default="torchvision")
    parser.add_argument("--distillation_method", type=str, choices=['TargetMatching'])
    parser.add_argument("--duo_mode", type=str, default="asymmetric", 
                       choices=["unweighted", "uqonly", "asymmetric"])
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--soft_target_loss_weight", type=float, default=1.0)
    parser.add_argument("--use_multi_transform", action="store_true", 
                       help="Use different transforms for teacher and student")
    
    args = parser.parse_args()
    
    print("🧾 Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_cls = IWildCamDataset if args.dataset_name.lower() == 'iwildcam' else Caltech256Dataset
    num_classes = dataset_cls.num_classes
    
    # Load student model first to get transforms
    student, student_transforms = get_model_with_head(
        model_name=args.student_name,
        num_classes=num_classes,
        source=args.student_source,
    )
    student.to(device)
    student.load_state_dict(torch.load(args.student_path, map_location=device))
    
    # Create validation loader for temperature calibration
    val_dataloaders = get_dataloaders(args.dataset_name, args.dataset_dir,
                                    batch_size=32, num_workers=4, 
                                    transforms=student_transforms)
    val_loader = val_dataloaders["val"] if "val" in val_dataloaders else val_dataloaders["test"]
    
    # Create DuoWrapper teacher with calibration
    duo_teacher, teacher_transforms = create_duo_teacher_with_calibration(
        args.teacher_fl_name, args.teacher_fl_path, args.teacher_fl_source,
        args.teacher_fs_name, args.teacher_fs_path, args.teacher_fs_source,
        num_classes, device, val_loader, mode=args.duo_mode
    )
    
    # Create training dataloader
    if args.use_multi_transform:
        # Use different transforms for teacher and student
        transforms_dict_train = {
            'teacher_fl': teacher_transforms[0]["train"],
            'teacher_fs': teacher_transforms[1]["train"],
            'student': student_transforms["train"]
        }
        
        train_loader = create_multi_transform_dataloader(
            args.dataset_name, args.dataset_dir, transforms_dict_train,
            batch_size=32, num_workers=4, split='train'
        )
        
        val_loader = create_multi_transform_dataloader(
            args.dataset_name, args.dataset_dir, {'student': student_transforms["test"]},
            batch_size=32, num_workers=4, split='val'
        )
        
        test_loader = create_multi_transform_dataloader(
            args.dataset_name, args.dataset_dir, {'student': student_transforms["test"]},
            batch_size=32, num_workers=4, split='test'
        )
    else:
        # Use same transforms (student transforms)
        dataloaders = get_dataloaders(args.dataset_name, args.dataset_dir,
                                    batch_size=32, num_workers=4, 
                                    transforms=student_transforms)
        train_loader = dataloaders["train"]
        test_loader = dataloaders["test"]
    
    if args.distillation_method == "TargetMatching":
        tm = TargetMatching(f"{args.distillation_method}", duo_teacher, student, device, 
                          temperature=args.temperature)
        print(f"Starting training with DuoWrapper in {args.duo_mode} mode...")
        if args.duo_mode == "asymmetric":
            print(f"Using calibrated temperatures: Tl={duo_teacher.temp_large:.4f}, Ts={duo_teacher.temp_small:.4f}")
        pre_distillation_result = evaluate_with_temp_scaling(tm.student,val_loader, test_loader, device)
        print(pre_distillation_result)
        tm.train(args.epochs, args.lr, train_loader, args.soft_target_loss_weight)
        result = evaluate_with_temp_scaling(tm.distilled_student,val_loader, test_loader, device)
        print(result)
    else:
        raise NotImplementedError(f"Unknown Distillation Method encountered: {args.distillation_method}")

if __name__ == "__main__":
    main()