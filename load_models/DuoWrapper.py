import torch
import torch.nn as nn
import torch.nn.functional as F
from load_models.TempScaleWrapper import TempScaleWrapper

def compute_ece(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            bin_accuracy = accuracies[mask].float().mean()
            bin_confidence = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_confidence - bin_accuracy)
    return ece.item()

def softmax_kl(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")

class DuoWrapper(nn.Module):
    def __init__(self, model_large: nn.Module, model_small: nn.Module, mode="softvote"):
        super().__init__()
        assert mode in ["softvote", "dictatorial", "confident", "weighted_voting"]
        self.mode = mode

        if mode == "weighted_voting":
            # Remove TempScaleWrapper if already applied
            if isinstance(model_large, TempScaleWrapper):
                print("⚠️ Unwrapping model_large for fresh temperature calibration.")
                model_large = model_large.model
            if isinstance(model_small, TempScaleWrapper):
                print("⚠️ Unwrapping model_small for fresh temperature calibration.")
                model_small = model_small.model

        self.model_large = model_large
        self.model_small = model_small

    def forward(self, x):
        """Just return logits for loss computation or evaluation."""
        logit_l = self.model_large(x)
        logit_s = self.model_small(x)
        return (logit_l + logit_s) / 2  # [B, C]

    def predict_with_uncertainty(self, x):
        """Return detailed predictions and uncertainty scores."""
        logit_l = self.model_large(x)
        logit_s = self.model_small(x)
        avg_logit = (logit_l + logit_s) / 2

        probs_avg = F.softmax(avg_logit, dim=-1)
        preds = probs_avg.argmax(dim=-1)
        cert_avg = probs_avg.max(dim=-1).values

        if self.mode in {"softvote","weighted_voting"}:
            result = {
                "logit": avg_logit,
                "probs": probs_avg,
                "preds": preds,
                "uncertainty(softmax_response)": 1 - cert_avg,
            }
        elif self.mode == "dictatorial":
            probs_l = F.softmax(logit_l, dim=-1)
            result={
                "logit": logit_l,
                "probs": probs_l,
                "preds": logit_l.argmax(dim=-1),
                "uncertainty(softmax_response)": 1 - cert_avg,
            }
        elif self.mode == "confident":
            # Choose the more certain model per sample
            cert_l = F.softmax(logit_l, dim=-1).max(dim=-1).values
            cert_s = F.softmax(logit_s, dim=-1).max(dim=-1).values
            mask = cert_l > cert_s
            chosen_logits = torch.where(mask[:, None], logit_l, logit_s)
            chosen_probs = F.softmax(chosen_logits, dim=-1)
            chosen_cert = chosen_probs.max(dim=-1).values
            result={
                "logit": chosen_logits,
                "probs": chosen_probs,
                "preds": chosen_logits.argmax(dim=-1),
                "uncertainty(softmax_response)": 1 - chosen_cert
            }

        return result

    def jointly_calibrate_temperature(self, logits_l, logits_s, labels):
        print("🎯 Joint temperature calibration in progress...")
        best_nll = float("inf")
        best_Tl, best_Ts = 1.0, 1.0

        for Tl in torch.arange(0.05, 5.05, 0.2):
            for Ts in torch.arange(0.05, 5.05, 0.2):
                logits_avg = (logits_l / Tl + logits_s / Ts) / 2
                nll = F.cross_entropy(logits_avg, labels).item()
                if nll < best_nll:
                    best_nll = nll
                    best_Tl, best_Ts = Tl.item(), Ts.item()

        print(f"Grid best Tl={best_Tl:.2f}, Ts={best_Ts:.2f}, NLL={best_nll:.4f}")

        Tl = torch.tensor([best_Tl], requires_grad=True, device=logits_l.device)
        Ts = torch.tensor([best_Ts], requires_grad=True, device=logits_s.device)
        optimizer = torch.optim.LBFGS([Tl, Ts], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            logits_avg = (logits_l / Tl + logits_s / Ts) / 2
            loss = F.cross_entropy(logits_avg, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        final_Tl, final_Ts = Tl.item(), Ts.item()
        print(f"Refined Tl={final_Tl:.4f}, Ts={final_Ts:.4f}")
        print(f"Final NLL = {F.cross_entropy((logits_l / Tl + logits_s / Ts)/2, labels).item():.4f}")

        self.model_large = TempScaleWrapper(self.model_large, init_temp=final_Tl)
        self.model_small = TempScaleWrapper(self.model_small, init_temp=final_Ts)
        print("✅ Joint calibration complete and models wrapped.")
