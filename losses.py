import torch
import torch.nn as nn
import torch.nn.functional as F

cross_entropy = nn.CrossEntropyLoss()


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=4.0,
    alpha=0.5,
):
    """
    Geoffrey Hinton's Knowledge Distillation Loss

    Parameters
    ----------
    student_logits : output from student model
    teacher_logits : output from teacher model
    labels         : ground-truth labels
    temperature    : controls softness of probability distribution
    alpha          : weight between hard and soft losses
    """

    # Hard Label Loss
    hard_loss = cross_entropy(student_logits, labels)

    # Teacher Soft Targets
    teacher_probs = F.softmax(
        teacher_logits / temperature,
        dim=1
    )

    # Student Soft Predictions
    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=1
    )

    # KL Divergence
    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    )

    # Final Distillation Loss
    loss = (
        alpha * hard_loss
        +
        (1 - alpha)
        * (temperature ** 2)
        * soft_loss
    )

    return loss