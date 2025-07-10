# DistillMethod.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.uncertainty_metrics import evaluate_model

class DistillMethod:
    def __init__(self, method_name, teacher, student, device):
        self.method_name = method_name
        self.teacher = teacher
        self.student = student
        self.distilled_student = None
        self.device = device
    
    def __str__(self):
        return f"{self.method_name} with {self.teacher} as teacher and {self.student} as student"
    
    def train(self):
        raise NotImplementedError("A distill method should implement its own training method")
    
    def evaluate(self, test_loader):
        model_to_evaluate = self.distilled_student if self.distilled_student is not None else self.student
        results = evaluate_model(model_to_evaluate, test_loader, self.device, distill=True)
        return results
