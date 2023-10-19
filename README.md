# cMBDF
convolutional MBDF via fast fourier transforms

Usage :

```
from cMBDF import get_convolutions, get_cmbdf

convs = get_convolutions() #only needs to be done once, can also store it as .npy 

pad_size = max([len(mol.charges) for mol in mols])
rep, drep = [], []
for mol in mols:
    r, dr = get_cmbdf(mol.charges,mol.coordinates,convs,pad_size,gradients=True)
    rep.append(r)
    drep.append(dr)
```
replace `mol.charges` and `mol.coordinates` with however you get those from the `mol` object
