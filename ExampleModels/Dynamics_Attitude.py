import torch

# Maps and gradients
# All vectors here are and must be column vectors.
# All tensors here: vectors and arrays must be 2d tensors.
# Given rx: dim of x, ry: dim of y.

rx = 6
ru = 3
ry = 3

def stateDynamics(x, u):
    x = torch.atleast_1d(x.squeeze())
    u = torch.atleast_1d(u.squeeze())
    tau = 0.05
    w = x[0:3]
    I = x[3:6]
    w_dot = torch.zeros(3,1)
    w_dot[0,:] = (u[0] - (I[2]-I[1])*w[1]*w[2]) / I[0]
    w_dot[1,:] = (u[1] - (I[0]-I[2])*w[2]*w[0]) / I[1]
    w_dot[2,:] = (u[2] - (I[1]-I[0])*w[0]*w[1]) / I[2]

    w_plus = w + tau * w_dot.squeeze()
    w_plus = w_plus % (2 * torch.pi) - torch.pi
    w_plus = torch.atleast_2d(w_plus)
    I = torch.atleast_2d(I)
    return torch.cat((w_plus, I), dim=1).T

def measurementDynamics(x, u):
    x = torch.atleast_1d(x.squeeze())
    u = torch.atleast_1d(u.squeeze())
    gx = x[0:3]
    return torch.atleast_2d(gx.squeeze()).T

# We follow control theory notation for compactness: f is the stateDynamics function, g is the measurementDynamics function
def f_Jacobian(x, u):
    f_x, _ = torch.autograd.functional.jacobian(stateDynamics, inputs=(x, u))
    return torch.atleast_2d(f_x.squeeze())
def g_Jacobian(x, u):
    g_x, _ = torch.autograd.functional.jacobian(measurementDynamics, inputs=(x, u))
    return torch.atleast_2d(g_x.squeeze())