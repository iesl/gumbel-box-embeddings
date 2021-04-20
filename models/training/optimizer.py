import torch
from torch.optim.optimizer import required
from allennlp.training.optimizers import Optimizer
from ..utils.mpe_utils import *

def euclidean_update(p, d_p, lr):
    p.data = p.data - lr * d_p
    return p.data

def poincare_grad(p, d_p):
    p_sqnorm = torch.clamp(torch.sum(p.data ** 2, dim=-1, keepdim=True), 0, 1-1e-5)
    d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p

def poincare_update(p, d_p, lr):
    v = -lr * d_p
    p.data = full_p_exp_map(p.data, v)
    return p.data

@Optimizer.register("riemann-sgd")
class RiemannianSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(RiemannianSGD, self).__init__(params, defaults)        

    def step(self, lr=None):
        loss = None
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group["lr"]
                if 'flag' in group and group['flag'] == 1:
                    d_p = poincare_grad(p, d_p)
                    p.data = poincare_update(p, d_p, lr)
                else:
                    p.data = euclidean_update(p, d_p, lr)
        return loss