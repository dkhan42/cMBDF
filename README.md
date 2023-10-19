# cMBDF
convolutional MBDF via fast fourier transforms

Usage :

```
from cMBDF import get_convolutions, get_cmbdf

convs = get_convolutions(gradients=True) #only needs to be done once, can also store it as .npy 

pad_size = max([len(mol.charges) for mol in mols]) #if not provided, defaults to molecule size with no padding
rep, drep = [], []
for mol in mols:
    r, dr = get_cmbdf(mol.charges,mol.coordinates,convs,pad_size,gradients=True)
    rep.append(r)
    drep.append(dr)
```
replace `mol.charges` and `mol.coordinates` with however you get those from the `mol` object

Usage (joblib):
```
from cMBDF_joblib import get_cmbdf
reps, dreps = cMBDF_joblib.generate_mbdf(mols_charges, mols_coords, gradients=True, progress_bar = True/False)
```
`mols_charges` and `mols_coords` should contain charges and coordinates arrays for multiple molecules to make this efficient.
