import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_ece(probs, labels, n_bins=15):
        """Computes Expected Calibration Error (ECE)."""
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
    
class TempScaleWrapper(nn.Module):
    def __init__(self, model: nn.Module, init_temp: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, x):
        logits = self.model(x)
        logits = logits["logit"] if isinstance(logits, dict) else logits
        return logits / self.temperature.to(logits.device)

    def calibrate_temperature(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Calibrates temperature using grid search + LBFGS refinement."""
        # Fasr grid search for temperature
        original_probs = F.softmax(logits, dim=-1)
        original_nll = F.cross_entropy(logits, labels).item()
        original_ece = compute_ece(original_probs, labels)
        print(f"NLL before temperature scaling = {original_nll:.4f}")
        print(f"ECE before temperature scaling = {original_ece:.4f}")
        best_nll = float("inf")
        best_T = 1.0

        for T in torch.arange(0.05, 5.05, 0.05):
            T = T.item()
            loss = F.cross_entropy(logits / T, labels).item()
            if loss < best_nll:
                best_nll = loss
                best_T = T
        print(f"Grid search best T = {best_T:.3f}, NLL = {best_nll:.4f}")
        
        # Refine with LBFGS
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
        self.temperature.data.copy_(temp_tensor.data)

        final_nll = F.cross_entropy(logits / T_refined, labels).item()
        scaled_ece = compute_ece(F.softmax(logits / T_refined, dim=-1), labels)
        print(f"Refined T = {T_refined:.4f}")
        print(f"NLL after temperature scaling = {final_nll:.4f}")
        print(f"ECE after temperature scaling = {scaled_ece:.4f}")
        return T_refined
    
    def predict_with_uncertainty(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        certainty = probs.max(dim=-1).values
        return {
            "logit": logits,
            "probs": probs,
            "preds": preds,
            "uncertainty(softmax_response)": 1 - certainty
    }