# TargetMatching.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from DistillMethod import DistillMethod

class TargetMatching(DistillMethod):
    def __init__(self, method_name, teacher, student, device, temperature=3.0):
        super().__init__(method_name, teacher, student, device)
        self.temperature = temperature
    
    def train(self, epochs, lr, train_loader, soft_target_loss_weight=0.7):
        """
        Modified to work with a single dataloader that handles different transforms
        """
        if not (0 <= soft_target_loss_weight <= 1):
            raise ValueError(f"soft_target_loss_weight must be between 0 and 1, got {soft_target_loss_weight}")
        
        optimizer = optim.AdamW(self.student.parameters(), lr=lr)
        ce_loss = nn.CrossEntropyLoss()
        
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = 0
            
            for batch_data in train_loader:
                # Handle different batch formats
                if isinstance(batch_data, dict):
                    # Multi-transform batch format
                    inputs_student = batch_data['student_inputs'].to(self.device)
                    inputs_teacher_fl = batch_data['teacher_fl_inputs'].to(self.device)
                    inputs_teacher_fs = batch_data['teacher_fs_inputs'].to(self.device)
                    labels = batch_data['labels'].to(self.device)
                else:
                    # Standard batch format (inputs, labels)
                    inputs, labels = batch_data
                    inputs_student = inputs.to(self.device)
                    inputs_teacher_fl = inputs.to(self.device)  # Same inputs for both
                    inputs_teacher_fs = inputs.to(self.device)  # Same inputs for both
                    labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_logits = self.teacher({"teacher_fl_inputs":inputs_teacher_fl,
                                                   "teacher_fs_inputs":inputs_teacher_fs})
                
                student_logits = self.student(inputs_student)
                
                T = self.temperature
                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
                
                soft_targets_loss = nn.functional.kl_div(soft_prob, soft_targets, 
                                                       reduction='batchmean') * (T ** 2)
                label_loss = ce_loss(student_logits, labels)
                
                loss = soft_target_loss_weight * soft_targets_loss + \
                       (1 - soft_target_loss_weight) * label_loss
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
            
            avg_loss = running_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.distilled_student = self.student
        return self.student