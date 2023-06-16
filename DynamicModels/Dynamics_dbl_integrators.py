import torch

# Maps and gradients
# All vectors here are and must be column vectors.
# All tensors here: vectors and arrays must be 2d tensors.
# Given rx: dim of x, ry: dim of y.

rx = 6
ru = 1
ry = 1


def stateDynamics(x,u):
    x = torch.atleast_1d(x.squeeze())
    theta = x[2:6]
    u = torch.atleast_1d(u.squeeze())
    f = torch.zeros(rx,)
    f[0] = theta[0] * x[0] + theta[1] * x[1]
    f[1] = theta[2] * x[1] + theta[3] * u[0]
    f[2] = theta[0]
    f[3] = theta[1]
    f[4] = theta[2]
    f[5] = theta[3]
    return torch.atleast_2d(f.squeeze()).T


def measurementDynamics(x, u):
    x = torch.atleast_1d(x.squeeze())
    u = torch.atleast_1d(u.squeeze())
    gx = torch.zeros(ry,)
    gx[0] = 2 * torch.tanh((x[0]-2)) + x[1] #1/9 * x[0] ** 3
    return torch.atleast_2d(gx.squeeze()).T

# We follow control theory notation for compactness: f is the stateDynamics function, g is the measurementDynamics function
def f_Jacobian(x, u):
    f_x, _ = torch.autograd.functional.jacobian(stateDynamics, inputs=(x, u))
    return torch.atleast_2d(f_x.squeeze())
def g_Jacobian(x, u):
    g_x, _ = torch.autograd.functional.jacobian(measurementDynamics, inputs=(x, u))
    return torch.atleast_2d(g_x.squeeze())