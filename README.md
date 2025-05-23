# cMBDF
convolutional MBDF via fast fourier transforms


Usage similar to original MBDF (joblib) for entire dataset:
```
from cMBDF import generate_mbdf
reps = generate_mbdf(mols_charges, mols_coords, progress_bar=True) #to get a tqdm progress bar
reps_global = generate_mbdf(mols_charges, mols_coords, local=False) #to get flattened feature vectors
```
`mols_charges` and `mols_coords` should contain charges and coordinates arrays for multiple molecules to make this efficient.

Usage (on the fly) per molecule :

```
from cMBDF import get_convolutions, get_cmbdf

convs = get_convolutions(gradients=True) #only needs to be done once since the convolutions are unique, can also store it as .npy 

pad_size = max([len(q) for q in mols_charges]) #if not provided, defaults to molecule size with no padding
rep_list = [get_cmbdf(q,r,convs,pad_size) for q,r in zip(mols_charges, mols_coords)]
```
Note that the `get_cmbdf` function does not use parallelization

# References
Please cite following work :

1. Danish Khan, O. Anatole von Lilienfeld; Generalized convolutional many body distribution functional representations. arXiv:2409.20471


2. Danish Khan, Stefan Heinen, O. Anatole von Lilienfeld; Kernel based quantum machine learning at record rate: Many-body distribution functionals as compact representations. J. Chem. Phys. 21 July 2023; 159 (3): 034106. https://doi.org/10.1063/5.0152215
