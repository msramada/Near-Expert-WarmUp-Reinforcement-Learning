import torch

# Maps and gradients
# All vectors here are and must be column vectors.
# All tensors here: vectors and arrays must be 2d tensors.
# Given rx: dim of x, ry: dim of y.

rx = 1
ru = 1
ry = 1


def stateDynamics(x,u):
    x = torch.atleast_1d(x.squeeze())
    u = torch.atleast_1d(u.squeeze())
    fxu = torch.zeros(rx,)
    fxu[0] = 1.0 * x[0] + u[0]
    return torch.atleast_2d(fxu.squeeze()).T


def measurementDynamics(x, u):
    x = torch.atleast_1d(x.squeeze())
    u = torch.atleast_1d(u.squeeze())
    gx = torch.zeros(ry,)
    gx[0] = 1/9 * x[0] ** 3
    return torch.atleast_2d(gx.squeeze()).T

# We follow control theory notation for compactness: f is the stateDynamics function, g is the measurementDynamics function
def f_Jacobian(x, u):
    f_x, _ = torch.autograd.functional.jacobian(stateDynamics, inputs=(x, u))
    return torch.atleast_2d(f_x.squeeze())
def g_Jacobian(x, u):
    g_x, _ = torch.autograd.functional.jacobian(measurementDynamics, inputs=(x, u))
    return torch.atleast_2d(g_x.squeeze())