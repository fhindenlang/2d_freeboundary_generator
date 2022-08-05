# 2D free-boundary generator
This repo sets up a 2D testcase using python to test the Biest code for the "Merkel" free-boundary problem :

`B.n=0`  with `B=grad phi + B0`

where `phi` is the unknown on a closed surface and `B0` is given (very hand-wavy here).

## setup of the testcase

The setup of the testcase is done in the notebook `testcase_setup.ipynb`.

The case is 2D in the R,Z plane and the field of a set poloidal coils (defined in `coils_symmetric.txt`) 
is computed via `Coil_contribution.py` from point-sources of current and the kernel of the Grad-Shafranov equation (`GS_kernels.py`).

A ficticous "plasma" current is added to get a region of closed flux surfaces (contours of psi). 
Then we fit a curve to one flux surface, using Fourier series for the curve R,Z coordinates  to have a fully parametrized curve.

We define a second magnetic field `B0` having the same circulation as the previous field (this is important since phi is periodic on the closed surface)
but the current is placed differently, such that `B0.n` is non-zero.

Then we want to compare the solution of Biest for `grad phi` to `B-B0`.


