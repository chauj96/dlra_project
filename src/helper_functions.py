# helper_functions contains RK3 ODE solver MFEM coefficients and sparse matrices

import numpy as np
from scipy import sparse
from scipy.io import savemat, loadmat
from scipy.sparse.linalg import splu
import mfem.ser as mfem


# time stepping one step ode solver
def rk3_ssp_step(f, t0, dt, y0):
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


# convert MFEM sparse matrix to scipy format
def mfem_sparse_to_csr(A):
    height = A.Height()
    width = A.Width()

    AD = A.GetDataArray().copy()
    AI = A.GetIArray().copy()
    AJ = A.GetJArray().copy()
    A = sparse.csr_matrix((AD, AJ, AI), shape=(height, width))
    return A

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