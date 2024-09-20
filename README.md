# cMBDF
convolutional MBDF via fast fourier transforms


Usage similar to original MBDF (joblib):
```
from cMBDF import generate_mbdf
reps = generate_mbdf(mols_charges, mols_coords)
reps_global = generate_mbdf(mols_charges, mols_coords, local=False) #to get flattened feature vectors
```
`mols_charges` and `mols_coords` should contain charges and coordinates arrays for multiple molecules to make this efficient.

Usage (on the fly with gradients) :

```
from cMBDF import get_convolutions, get_cmbdf

convs = get_convolutions(gradients=True) #only needs to be done once since the convolutions are unique, can also store it as .npy 

pad_size = max([len(q) for q in mols_charges]) #if not provided, defaults to molecule size with no padding
rep, drep = [], []
for q,r in zip(mols_charges, mols_coords):
    r, dr = get_cmbdf(q,r,convs,pad_size,gradients=True)
    rep.append(r)
    drep.append(dr)
```
Note that there is currently some problem in the `get_cmbdf` function which does not allow using `Numba`'s `prange` parallelization option.
Hence I provide the `generate_mbdf` function which uses `joblib`'s parallelisation. A fix would be appreciated!
