# Dynamical low-rank approximation of the Vlasov-Poisson equation
Landau damping simulation for the dynamical low-rank approximation (DLRA) with 
1) projector splitting integrator
2) rank-adaptive integrator.

## Build MFEM with Python wrapper PyMFEM
To execute the code, the MFEM library along with its Python Wrapper PyMFEM needs to be built and properly configured on our system.
Please refer to the following website and ensure that we have a functional build system comprising c++, make and cmake.
Additionally, we assume Anaconda is the Python distribution being utilized. All necessary procedures are detailed within the file name 	`pymfem/build_pymfem-4.6.1.0.sh`.
* [MFEM](https://mfem.org/)
* [PyMFEM](https://github.com/mfem/PyMFEM)

Here is stpes for initial setting:

Create the Anaconda Python environment
```
conda create -y -n pymfem-4.6.1.0 python=3.11 numpy scipy matplotlib cmake swig
conda activate pymfem-4.6.1.0
```
Build the MFEM library
```
git clone https://github.com/mfem/mfem.git
cd mfem
git checkout 4a45c70d1269d293266b77a3a025a9756d10ed8f  
git apply ../mfem-4.6.1.0.patch
mkdir mfem_build ; cd mfem_build
cmake .. -DBUILD_SHARED_LIBS=1 -DCMAKE_INSTALL_PREFIX="../mfem_install"
make -j 4
make install
cd ../..
```
Build PyMFEM
```
git clone https://github.com/mfem/PyMFEM.git
cd PyMFEM
git checkout v_4.6.1.0
python setup.py install --mfem-source="../mfem/mfem_build" --mfem-prefix="../mfem/mfem_install"
```
