# Code for 1D Landau damping simulation

import sys
import os
import shutil
import getopt
import numpy as np
import mfem.ser as mfem

from matrices import Mx, Tx, Txx, Mx1, Mx2, Mv, Tv, Tvv, Mv1, Mv2, Mv_squared, Mx_inv, Mv_inv
from helper_functions import orthogonalize, rungekutta_3, physical_quantities, X0_int, V0_int
from electric_field import Electric_Field
from scipy.io import savemat
from scipy.sparse.linalg import splu


# initial settings
exe_index = 0
refine = 0 
dt = 1e-3 
x_grid = 64
v_grid = 256
termination = 50
ode_step = rungekutta_3
rank = 3
time_method = 2  # 1: projector-splitting / 2:rank-adaptive
rauc_tau = 1e-4  
rauc_max = 10  
visualization = False
one = mfem.ConstantCoefficient(1.0)

# setting up time integration
nt = int(np.round(termination/dt))
dt = termination/nt

# set up mesh for periodic boundary
meshfilex0 = "landau_per_x_%i.mesh" % x_grid
meshfilev0 = "box_v_6.0_%i.mesh" % v_grid
meshx = mfem.Mesh("data/%s" % meshfilex0, 1, 1)

# x refinement
order = 1
for i in range(refine):
    meshx.UniformRefinement()
fecx = mfem.H1_FECollection(order,  meshx.Dimension())
fespacex = mfem.FiniteElementSpace(meshx, fecx)
rhox_gf = mfem.GridFunction(fespacex)

# v refinement
meshv = mfem.Mesh("data/%s" % meshfilev0, 1, 1)
for i in range(refine):
    meshv.UniformRefinement()
fecv = mfem.H1_FECollection(order,  meshv.Dimension())
fespacev = mfem.FiniteElementSpace(meshv, fecv)
rhov_gf = mfem.GridFunction(fespacev)

# compute the electric field
efield_solver = Electric_Field(meshx, 1)

# integration rules
int_order = order + 1
irs = [mfem.IntRules.Get(i, int_order) for i in range(mfem.Geometry.NumGeom)]

# interpolation of initial conditions
X0_grid = mfem.GridFunction(fespacex)
X0_grid.ProjectCoefficient(X0_int())
X0_1 = X0_grid.GetDataArray().copy()

nx = fespacex.GetNDofs()
X0 = np.zeros((nx, rank))
X0[:, 0] = X0_1

V0_grid = mfem.GridFunction(fespacev)
V0_grid.ProjectCoefficient(V0_int())
V0_1 = V0_grid.GetDataArray().copy()

nv = fespacev.GetNDofs()
V0 = np.zeros((nv, rank))
V0[:, 0] = V0_1

S0 = np.zeros((rank, rank))
S0[0, 0] = 1.0

# orthonormalize initial condition
X0, Sx = orthogonalize(X0, Mx)
V0, Sv = orthogonalize(V0, Mv)
S0 = Sx @ S0 @ Sv.T

# compute the electric field with respect to current density
def update_E(X, S, V):
    global ME1, ME2
    # compute density
    rhoV = np.ones((1, fespacev.GetNDofs())) @ (Mv @ V)
    rhox_gf.GetDataArray()[:] = (X @ S @ rhoV.T).reshape(-1)
    rhox_gf_coeff = mfem.GridFunctionCoefficient(rhox_gf)

    # solve for efield
    phi, E1_gf, E2_gf = efield_solver.solve(rhox_gf_coeff)

    # compute electrical energy as squared L2 norm of coefficients
    E1_data = E1_gf.GetDataArray()
    E2_data = E2_gf.GetDataArray()
    E_electric = 0.5 * (E1_data @ (efield_solver.Mx_L2 @ E1_data) + E2_data @ (efield_solver.Mx_L2 @ E2_data))

    # compute discretization matrices
    E1 = mfem.GridFunctionCoefficient(E1_gf)
    mE1 = mfem.BilinearForm(fespacex)
    mE1.AddDomainIntegrator(mfem.MassIntegrator(E1))
    mE1.Assemble()
    mE1.Finalize()
    ME1 = mE1.SpMat()

    E2 = mfem.GridFunctionCoefficient(E2_gf)
    mE2 = mfem.BilinearForm(fespacex)
    mE2.AddDomainIntegrator(mfem.MassIntegrator(E2))
    mE2.Assemble()
    mE2.Finalize()
    ME2 = mE2.SpMat()

    return E_electric

# single time step for low-rank approximation
def step1(X, S, V, dt):
    def eval_rhsK(tau, K, V):
        global ME1, ME2
        rhsK = (
            - (Tx @ K) @ (V.T @ (Mv1 @ V)).T
            - (Txx @ K) @ (V.T @ (Mv2 @ V)).T
            + (ME1 @ K) @ (V.T @ (Tv @ V)).T
            + (ME2 @ K) @ (V.T @ (Tvv @ V)).T)

        dK = Mx_inv.solve(rhsK)
        return dK

    def eval_rhsS(tau, X, S, V):
        global ME1, ME2
        dS = (
            - (X.T @ (Tx @ X)) @ S @ (V.T @ (Mv1 @ V)).T
            - (X.T @ (Txx @ X)) @ S @ (V.T @ (Mv2 @ V)).T
            + (X.T @ (ME1 @ X)) @ S @ (V.T @ (Tv @ V)).T
            + (X.T @ (ME2 @ X)) @ S @ (V.T @ (Tvv @ V)).T)
        return dS

    def eval_rhsL(tau, L, X):
        global ME1, ME2
        rhsL = (
            - (Mv1 @ L) @ (X.T @ (Tx @ X)).T
            - (Mv2 @ L) @ (X.T @ (Txx @ X)).T
            + (Tv @ L) @ (X.T @ (ME1 @ X)).T
            + (Tvv @ L) @ (X.T @ (ME2 @ X)).T)
        dL = Mv_inv.solve(rhsL)
        return dL


    # core step
    match time_method:
        # projector-splitting time integrator
        case 1:
            # K step
            K = X @ S
            K = ode_step(lambda tau, Y: eval_rhsK(tau, Y, V), 0.0, dt, K)
            X, S = orthogonalize(K, Mx)

            # S step
            S = ode_step(lambda tau, Y: -eval_rhsS(tau, X, Y, V), 0.0, dt, S)

            # L step
            L = V @ S.T
            L = ode_step(lambda tau, Y: eval_rhsL(tau, Y, X), 0.0, dt, L)
            V, S = orthogonalize(L, Mv)
            S = S.T

        case 2:
            # rank-adaptive time integrator
            currank = X.shape[1]

            # K step
            K = X @ S
            K1 = ode_step(lambda tau, Y: eval_rhsK(tau, Y, V), 0.0, dt, K)
            Xhat, Sx = orthogonalize(np.hstack((X, K1)), Mx)
            Rx = Sx[:, :currank]

            # L step
            L = V @ S.T
            L1 = ode_step(lambda tau, Y: eval_rhsL(tau, Y, X), 0.0, dt, L)
            Vhat, Sv = orthogonalize(np.hstack((V, L1)), Mv)
            Rv = Sv[:, :currank]

            # S step
            Shat = Rx @ S @ Rv.T
            Shat1 = ode_step(lambda tau, Y: eval_rhsS(tau, Xhat, Y, Vhat),
                             0.0, dt, Shat)

            # truncation / augmentation
            Qx, sigma, QvT = np.linalg.svd(Shat1)
            Qv = QvT.T

            sigma_flipped = np.flip(sigma)
            mask = np.cumsum(sigma_flipped ** 2) < rauc_tau ** 2
            kinv = np.argmin(mask)
            if mask[kinv] == True:  ## all values smaller than rauc_tau
                kinv = 2 * currank
            newrank = max(1, min(2 * currank - kinv, rauc_max))

            # reduction of rank
            S = np.diag(sigma[:newrank])
            X = Xhat @ Qx[:, :newrank]
            V = Vhat @ Qv[:, :newrank]

    return (X, S, V)

# time stepping
X = X0.copy()
S = S0.copy()
V = V0.copy()

t = 0.0

# update the electric field
E_electric = update_E(X, S, V)

# compute initial physical quantities
mass_0, E_total_0, entropy_0 = physical_quantities(X, S, V, E_electric)

for i in range(nt):
    (X, S, V) = step1(X, S, V, dt)
    E_electric = update_E(X, S, V)
    t = (i+1) * dt

    # update physical quantities and compute relative error of the quantities
    mass, E_total, entropy = physical_quantities(X, S, V, E_electric)
    relerr_mass = np.abs((mass-mass_0)/mass_0)
    relerr_E_total = np.abs((E_total-E_total_0)/E_total_0)
    relerr_entropy = np.abs((entropy-entropy_0)/entropy_0)
