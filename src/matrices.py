from scipy.sparse.linalg import splu
import mfem.ser as mfem
import numpy as np

from helper_functions import sum_squared

# mass matrix and assembly
mx = mfem.BilinearForm(fespacex)
mx.AddDomainIntegrator(mfem.MassIntegrator(one))
mx.Assemble()
mx.Finalize()
Mx = mx.SpMat()

# transport matrix and assembly
tx = mfem.BilinearForm(fespacex)
tx.AddDomainIntegrator(mfem.DerivativeIntegrator(one, 0))
tx.Assemble()
tx.Finalize()
Tx = tx.SpMat()

txx = mfem.BilinearForm(fespacex)
txx.AddDomainIntegrator(mfem.DerivativeIntegrator(one, 1))
txx.Assemble()
txx.Finalize()
Txx = txx.SpMat()

# inner product
rule = mfem.IntRules.Get(2, 4)
mx1 = mfem.BilinearForm(fespacex)
mx1.AddDomainIntegrator(mfem.MassIntegrator(x1(), rule))
mx1.Assemble()
mx1.Finalize()
Mx1 = mx1.SpMat()

mx2 = mfem.BilinearForm(fespacex)
mx2.AddDomainIntegrator(mfem.MassIntegrator(x2(), rule))
mx2.Assemble()
mx2.Finalize()
Mx2 = mx2.SpMat()

# mass matrix and assembly
mv = mfem.BilinearForm(fespacev)
mv.AddDomainIntegrator(mfem.MassIntegrator(one))
mv.Assemble()
mv.Finalize()
Mv = mv.SpMat()

# transport matrix and assembly
tv1 = mfem.BilinearForm(fespacev)
tv1.AddDomainIntegrator(mfem.DerivativeIntegrator(one, 0))
tv1.Assemble()
tv1.Finalize()
Tv = tv1.SpMat()

tv2 = mfem.BilinearForm(fespacev)
tv2.AddDomainIntegrator(mfem.DerivativeIntegrator(one, 1))
tv2.Assemble()
tv2.Finalize()
Tvv = tv2.SpMat()

# < v_i psi, psi>
rule = mfem.IntRules.Get(2, 4)
mv1 = mfem.BilinearForm(fespacev)
mv1.AddDomainIntegrator(mfem.MassIntegrator(x1(), rule))
mv1.Assemble()
mv1.Finalize()
Mv1 = mv1.SpMat()

mv2 = mfem.BilinearForm(fespacev)
mv2.AddDomainIntegrator(mfem.MassIntegrator(x2(), rule))
mv2.Assemble()
mv2.Finalize()
Mv2 = mv2.SpMat()

# < |v| 2 psi, psi>
rule = mfem.IntRules.Get(2, 4)
mv_squared = mfem.BilinearForm(fespacev)
mv_squared.AddDomainIntegrator(mfem.MassIntegrator(sum_squared(), rule))
mv_squared.Assemble()
mv_squared.Finalize()
Mv_squared = mv_squared.SpMat()

# lu decomposition
Mx_inv = splu(Mx)
Mv_inv = splu(Mv)