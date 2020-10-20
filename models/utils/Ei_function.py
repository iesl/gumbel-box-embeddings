import torch
from scipy import special

class ExpEi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        with torch.no_grad():
            x = torch.Tensor(special.expi(input.detach()))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output*(torch.exp(input)/input)
        return grad_input