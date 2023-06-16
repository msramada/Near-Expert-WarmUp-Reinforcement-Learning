import torch

# Maps and gradients
# All vectors here are and must be column vectors.
# All tensors here: vectors and arrays must be 2d tensors.
# Given rx: dim of x, ry: dim of y.
rx = 3
ru = 1
ry = 1

def stateDynamics(x,u):
    x = x.squeeze()
    u = u.squeeze()
    f = torch.zeros(rx,1)
    tau = 0.05
    v = 1/tau * 2
    u = 1/tau * u/100 #per second to per tau
    f[0] = x[0] + tau * v * torch.sin(u * tau / 2) / (u * tau / 2) * torch.cos(x[2] + u * tau / 2)
    f[1] = x[1] + tau * v * torch.sin(u * tau / 2) / (u * tau / 2) * torch.sin(x[2] + u * tau / 2)
    f[2] = x[2] + tau * u
    f[2] = f[2] % (2*torch.pi)
    return torch.atleast_2d(f.squeeze()).T


def measurementDynamics(x, u):
    x = x.squeeze()
    u = u.squeeze()
    gx = torch.zeros(ry, 1)
    gx[0] = 50 * torch.tanh(0.2 * (x[0]-5) ) * torch.tanh(0.2 * (x[1]-5) )
    #gx[1] = 5 * torch.tanh(0.5 * (x[1]-5) )
    return torch.atleast_2d(gx.squeeze()).T

# We follow control theory notation for compactness: f is the stateDynamics function, g is the measurementDynamics function
def f_Jacobian(x, u):
    f_x, _ = torch.autograd.functional.jacobian(stateDynamics, inputs=(x, u))
    return torch.atleast_2d(f_x.squeeze())
def g_Jacobian(x, u):
    g_x, _ = torch.autograd.functional.jacobian(measurementDynamics, inputs=(x, u))
    return torch.atleast_2d(g_x.squeeze())