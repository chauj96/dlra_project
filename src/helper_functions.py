# helper_functions contains RK3 ODE solver MFEM coefficients and sparse matrices

import mfem.ser as mfem
import numpy as np
from scipy import sparse
from scipy.io import savemat, loadmat
from scipy.sparse.linalg import splu
from matrices import Mx, Mv, Mv_squared



# time stepping one step ode solver
def rungekutta_3(f, t0, dt, y0):
    k0 = dt * f(t0, y0)
    y1 = y0 + k0
    k1 = dt * f(t0 + dt, y1)
    y2 = 0.75 * y0 + 0.25 * (y1 + k1)
    k2 = dt * f(t0 + 0.5*dt, y2)
    y3 = (1.0/3.0) * y0 + (2.0/3.0) * (y2 + k2)
    return y3


# modified Gram Schmidt orthogonalization
def orthogonalize(K, M):
    [V, S] = np.linalg.qr(K)

    m, n = V.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        R[j, j] = np.sqrt(np.dot(V[:, j], M @ V[:, j]))
        Q[:, j] = V[:, j] / R[j, j]
        for k in range(j + 1, n):
            R[j, k] = Q[:, j].T @ (M @ V[:, k])
            V[:, k] -= R[j, k] * Q[:, j]
    S = R @ S

    return (Q, S)


# return difference of two tensors in tensor format
def tensor_diff(X1, S1, V1, X2, S2, V2):
    r1 = S1.shape[0]
    r2 = S2.shape[0]
    r = r1 + r2

    X = np.zeros((X1.shape[0], r))
    X[:, :r1] = X1
    X[:, r1:] = X2

    S = np.zeros((r, r))
    S[:r1, :r1] = S1
    S[r1:, r1:] = -S2

    V = np.zeros((V1.shape[0], r))
    V[:, :r1] = V1
    V[:, r1:] = V2

    return (X, S, V)

# compute L2 of tensor with respect to mass matrices
def L2_norm(X, S, V, Mx, Mv):
    Ax = X.T @ (Mx @ X)
    Av = V.T @ (Mv @ V)
    return np.sqrt(np.sum((Ax @ S @ Av) * S))

# computing major physical quantities
def physical_quantities(X, S, V, E_electric):
    # mass
    rhoV = np.ones((1, fespacev.GetNDofs())) @ (Mv @ V)
    mass = np.sum(Mx @ (X @ S @ rhoV.T).reshape(-1))

    # kinetic energy
    rhoV = np.ones((1, fespacev.GetNDofs())) @ (Mv_squared @ V)
    E_kinetic = 0.5 * np.sum(Mx @ (X @ S @ rhoV.T).reshape(-1))

    # total energy
    E_total = E_kinetic + E_electric

    entropy = L2_norm(X, S, V, Mx, Mv)**2
    return mass, E_total, entropy

# coefficient of MFEM returning first component x_1
class x1(mfem.PyCoefficient):
    def EvalValue(self, x):
        return x[0]

# coefficient of MFEM returning fist component x_2
class x2(mfem.PyCoefficient):
    def EvalValue(self, x):
        return x[1]

# coefficient of MFEM returning squared sum of components
class sum_squared(mfem.PyCoefficient):
    def EvalValue(self, x):
        return x[0]**2 + x[1]**2
    
# interpolation of initial conditions
class X0_int(mfem.PyCoefficient):
    def __init__(self):
        super().__init__()
        self.k = 0.5
        self.alpha = 1e-2

    def EvalValue(self, x):
        return 1.0 \
            + self.alpha * np.cos(self.k * x[0]) \
            + self.alpha * np.cos(self.k * x[1])

class V0_int(mfem.PyCoefficient):
    def EvalValue(self, v):
        return np.exp(-0.5 * (v[0]**2 + v[1]**2)) / (2.0 * np.pi)
