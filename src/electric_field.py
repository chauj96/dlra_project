# FEM solver for the electric field

import numpy as np
from scipy.sparse.linalg import splu
import mfem.ser as mfem


# solve Poisson equation
class Electric_Field:
    def __init__(self, mesh, order):
        self.mesh = mesh
        self.order = order + 1

        # finite element ansatz functions
        self.dim = self.mesh.Dimension()
        self.fec_H1 = mfem.H1_FECollection(self.order, self.dim)
        self.fec_L2 = mfem.L2_FECollection(self.order, self.dim)

        # fe spaces
        self.fespace_H1 = mfem.FiniteElementSpace(self.mesh, self.fec_H1)
        self.fespace_L2 = mfem.FiniteElementSpace(self.mesh, self.fec_L2)
        self.fespace_L2_vec = mfem.FiniteElementSpace(
            self.mesh, self.fec_L2, self.dim, mfem.Ordering.byNODES)

        # Stiffness matrix for H1 elements
        one = mfem.ConstantCoefficient(1.0)
        sx = mfem.BilinearForm(self.fespace_H1)
        sx.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
        sx.Assemble()
        sx.Finalize()
        self.Sx = sx.SpMat()

        # mass matrix for L2 elements
        mx_L2 = mfem.BilinearForm(self.fespace_L2)
        mx_L2.AddDomainIntegrator(mfem.MassIntegrator(one))
        mx_L2.Assemble()
        mx_L2.Finalize()
        self.Mx_L2 = mx_L2.SpMat()

        # set dof zero as dirichlet node -> rest is free
        ndofs = self.fespace_H1.GetNDofs()
        self.index_free = np.arange(1, ndofs, dtype=np.int64)

        # lu decomposition of free part of stiffness matrix
        Sx_free = self.Sx[self.index_free, :][:, self.index_free]
        self.Sx_free_inv = splu(Sx_free)

        # prepare gridfunctions to store potential and gradient
        self.phi = mfem.GridFunction(self.fespace_H1)  # potential
        self.rho = mfem.GridFunction(self.fespace_H1)  # corrected density
        self.gradphi = mfem.GridFunction(self.fespace_L2_vec)  # full gradient
        self.Ex = mfem.GridFunction(self.fespace_L2)  # d/dx part of grad
        self.Ey = mfem.GridFunction(self.fespace_L2)  # d/dy part of grad
        return

    # solve poisson equation with rhs given as coefficient of MFEM
    def solve(self, rho_coeff):
        phi_data = self.phi.GetDataArray()

        # create coefficient for corrected density 1-rho
        one = mfem.ConstantCoefficient(1.0)
        rho_corr_coeff = mfem.SumCoefficient(one, rho_coeff, 1.0, -1.0)

        # compute rhs
        r = mfem.LinearForm(self.fespace_H1)
        r.AddDomainIntegrator(mfem.DomainLFIntegrator(rho_corr_coeff))
        r.Assemble()
        r_data = r.GetDataArray()

        # set dirichlet boundary node 0 to zero
        phi_data[:] = 0.0

        # compute solution from dirichlet and rhs (1.0-rho)
        b = r_data - (self.Sx @ phi_data)
        phi_data[self.index_free] = self.Sx_free_inv.solve(b[self.index_free])

        # compute gradient
        gradphi_gf = mfem.GradientGridFunctionCoefficient(self.phi)
        self.gradphi.ProjectCoefficient(gradphi_gf)
        gradphi_data = self.gradphi.GetDataArray()
        ndof = self.fespace_L2.GetNDofs()

        # E_x = - d/dx phi (first ndof coefficients in gradient)
        Ex_data = self.Ex.GetDataArray()
        Ex_data[:] = -gradphi_data[:ndof]

        # E_y = - d/dy phi (second ndof coefficients in gradient)
        Ey_data = self.Ey.GetDataArray()
        Ey_data[:] = -gradphi_data[ndof:]

        return (self.phi, self.Ex, self.Ey)
