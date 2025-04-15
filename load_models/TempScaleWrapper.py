import torch
import torch.nn as nn
import torch.nn.functional as F

class TempScaleWrapper(nn.Module):
    def __init__(self, model: nn.Module, init_temp: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, x):
        out = self.model(x)
        logits = out["logit"] if isinstance(out, dict) else out
        return {"logit": logits / self.temperature}

    def calibrate_temperature(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Calibrates temperature using grid search + LBFGS refinement."""
        # Fasr grid search for temperature
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

        print(f"Refined T = {T_refined:.4f}")
        return T_refined
    
    def predict_with_uncertainty(self, x):
        out = self.forward(x)
        logits = out["logit"]
        # Try to defer to inner model's uncertainty if available
        if hasattr(self.model, "predict_with_uncertainty"):
            return self.model.predict_with_uncertainty(x)
        # Fallback: compute softmax response as default
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        certainty = probs.max(dim=-1).values
        return {"logit": logits, 
                "probs": probs,
                "preds": preds,
                "uncertainty(softmax_response)": 1-certainty}
