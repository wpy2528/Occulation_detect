import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, device, epochs=0, temperature=5, alpha=0.5, loss=None):
        super().__init__()
        self.epochs = epochs
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.kl_loss.to(device)
        self.train_loss = loss if loss else nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels, epoch=0):
        if self.epochs != 0:
            tau = max(1.0, self.temperature, - 4.0 * (epoch / self.epochs))
        else:
            tau = self.temperature
        soft_teacher = F.softmax(teacher_logits / tau, dim=1)
        soft_student = F.log_softmax(student_logits / tau, dim=1)
        kl_loss = self.kl_loss(soft_student, soft_teacher) * (tau ** 2)
        train_loss = self.train_loss(student_logits, labels)

        return self.alpha * kl_loss + (1 - self.alpha) * train_loss, tau

