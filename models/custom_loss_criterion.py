import torch
import torch.nn as nn
from torch.autograd import Function

class CustomMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Ensure the input and target are the same shape
        return torch.mean((input - target) ** 2)



class CustomCosineLossFn(Function):
    @staticmethod
    def forward(ctx, input, target, eps=1e-8):
        ctx.original_input_shape = input.shape  # Save shape for backward

        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        input_norm = input.norm(dim=1, keepdim=True).clamp(min=eps)
        target_norm = target.norm(dim=1, keepdim=True).clamp(min=eps)

        input_normalized = input / input_norm
        target_normalized = target / target_norm

        cosine_sim = (input_normalized * target_normalized).sum(dim=1)
        loss = 1 - cosine_sim.mean()

        ctx.save_for_backward(input_normalized, target_normalized)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input_normalized, target_normalized = ctx.saved_tensors
        B, D = input_normalized.shape

        grad_input = -target_normalized / B
        grad_target = -input_normalized / B

        # Reshape to original input shape
        grad_input = grad_input.view(ctx.original_input_shape)
        grad_target = grad_target.view(ctx.original_input_shape)

        return grad_input * grad_output, grad_target * grad_output, None
    

class CustomCosineLoss(nn.Module):
    def forward(self, input, target):
        return CustomCosineLossFn.apply(input, target)


class HardThresholdLossFn(Function):
    @staticmethod
    def forward(ctx, input, target, threshold=0.01):
        ctx.save_for_backward(input, target)
        ctx.threshold = threshold
        loss = ((input - target) ** 2).mean()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        threshold = ctx.threshold

        diff = input - target
        mask = (diff ** 2) > threshold  # Only keep gradients where error is above threshold

        grad_input = (2 * diff / input.numel()) * mask
        grad_target = (-2 * diff / target.numel()) * mask

        return grad_input * grad_output, grad_target * grad_output, None  # None for threshold


# Wrapped loss
class HardThresholdLoss(nn.Module):
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = threshold

    def forward(self, input, target):
        return HardThresholdLossFn.apply(input, target, self.threshold)
    
