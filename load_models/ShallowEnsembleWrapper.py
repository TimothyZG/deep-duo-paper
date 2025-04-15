import torch.nn as nn
import torch.nn.functional as F

class ShallowEnsembleWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        logits = out["logit"] if isinstance(out, dict) else out  # [B, m_head, C]
        return {"logit": logits}

    def predict_with_uncertainty(self, x):
        out = self.forward(x)
        logits = out["logit"]  # [B, m_head, C]

        # Compute probabilities in log-space for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)         # [B, m_head, C]
        probs = log_probs.exp()                           # [B, m_head, C]

        mean_probs = probs.mean(dim=1)                    # [B, C]
        log_mean_probs = (mean_probs + 1e-8).log()        # avoid log(0)

        preds = mean_probs.argmax(dim=-1)                 # [B]
        certainty = mean_probs.max(dim=-1).values         # [B]

        # Mutual Information = H(E[p]) - E[H[p]]
        entropy_mean = -(mean_probs * log_mean_probs).sum(dim=-1)               # [B]
        entropy_per_head = -(probs * log_probs).sum(dim=-1)                     # [B, m_head]
        expected_entropy = entropy_per_head.mean(dim=1)                         # [B]
        mutual_info = entropy_mean - expected_entropy                           # [B]

        return {
            "logit": logits,
            "prob": mean_probs,
            "pred": preds,
            "uncertainty(softmax_response)": 1 - certainty,
            "uncertainty(mutual_information)": mutual_info
        }
